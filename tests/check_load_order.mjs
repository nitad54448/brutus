// check_load_order.mjs -- static check of the classic-script load order.
//
// Brutus's JavaScript is a set of classic scripts that share ONE global scope
// and run in the order brutus.html lists them (and, in the workers, the order
// js/crystallography/manifest.js lists them). That design has exactly one
// failure mode that no browser reports until the code path actually runs:
// code that executes while the scripts load (a top-level call, an initializer,
// a listener registered with a function value, an IIFE) touching a name that a
// LATER script defines -- or a `const` later in the same script (TDZ).
// That is how the old "WebGPU not available" warning broke: a startup call ran
// before `const showStatus` existed.
//
// This walks every top-level statement in load order, follows code that runs
// immediately (including the bodies of functions called at load time, IIFEs,
// forEach/map callbacks, constructors of classes instantiated at load time) and
// reports every reference to a top-level name that is not yet initialised.
// Callbacks handed to addEventListener, setTimeout, promises etc. are treated
// as deferred. It also checks that brutus.html and the manifest agree, that no
// name is declared twice in one context, and that no file in js/ is orphaned.
//
//   npm install --no-save typescript      (parser only; any recent version)
//   node check_load_order.mjs
//
// Exit code 0 = clean, 1 = problems found, 2 = could not run.
import { readFileSync, readdirSync, statSync } from 'fs';
import { join, relative } from 'path';
import { createRequire } from 'module';

const ROOT = new URL('.', import.meta.url).pathname;
let ts;
try {
    const require = createRequire(import.meta.url);
    ts = require(process.env.TYPESCRIPT_PATH || 'typescript');
} catch (_) {
    console.error('check_load_order.mjs needs the TypeScript parser:  npm install --no-save typescript');
    process.exit(2);
}

const read = (p) => readFileSync(join(ROOT, p), 'utf8').replace(/\r\n/g, '\n');

// ---- the three load orders -------------------------------------------------
const html = read('brutus.html');
const htmlScripts = [...html.matchAll(/<script\s+src="([^"?]+)(?:\?[^"]*)?"[^>]*>/g)]
    .map(m => m[1]).filter(p => p.startsWith('js/'));
const manifestSrc = read('js/crystallography/manifest.js');
const manifestFiles = [...manifestSrc.matchAll(/^\s*'([^']+\.js)',\s*$/gm)].map(m => 'js/crystallography/' + m[1]);

const contexts = {
    'main thread (brutus.html)': htmlScripts,
    'CPU index worker': ['js/crystallography/manifest.js', ...manifestFiles, 'js/workers/index-worker.js'],
    'refinement worker': ['js/crystallography/manifest.js', ...manifestFiles, 'js/workers/refinement-worker.js'],
};

const problems = [];
const problem = (msg) => problems.push(msg);

// brutus.html must load exactly the manifest's crystallography files, in order.
const htmlCryst = htmlScripts.filter(p => p.startsWith('js/crystallography/'));
if (JSON.stringify(htmlCryst) !== JSON.stringify(manifestFiles)) {
    problem(`brutus.html and js/crystallography/manifest.js list different crystallography files:\n` +
            `      html:     ${htmlCryst.join(', ')}\n      manifest: ${manifestFiles.join(', ')}`);
}

// Every .js under js/ must be loaded by something.
const walk = (dir) => readdirSync(join(ROOT, dir)).flatMap(n => {
    const p = join(dir, n);
    return statSync(join(ROOT, p)).isDirectory() ? walk(p) : (p.endsWith('.js') ? [p] : []);
});
const loaded = new Set(Object.values(contexts).flat());
for (const f of walk('js')) if (!loaded.has(f)) problem(`${f} is not loaded by brutus.html or by a worker`);

// Names that cannot be redeclared with let/const/class at global scope.
const RESTRICTED = new Set(['window', 'document', 'location', 'top', 'self', 'chrome', 'undefined', 'NaN', 'Infinity']);

// ---- analysis of one context ------------------------------------------------
const ARRAY_SYNC = new Set(['forEach', 'map', 'filter', 'some', 'every', 'reduce', 'reduceRight',
                            'find', 'findIndex', 'findLast', 'findLastIndex', 'flatMap', 'sort']);

function analyse(contextName, files) {
    for (const f of files) {
        try { statSync(join(ROOT, f)); } catch (_) { problem(`${contextName}: ${f} does not exist`); return; }
    }
    const host = ts.createCompilerHost({ allowJs: true });
    const getSourceFile = host.getSourceFile;
    host.getSourceFile = (name, lang) => name.startsWith(ROOT)
        ? ts.createSourceFile(name, readFileSync(name, 'utf8'), lang, true, ts.ScriptKind.JS)
        : getSourceFile(name, lang);
    const program = ts.createProgram(files.map(f => join(ROOT, f)),
        { allowJs: true, checkJs: false, noEmit: true, target: ts.ScriptTarget.ES2022, lib: ['lib.es2022.d.ts'], types: [] }, host);
    const checker = program.getTypeChecker();
    const sources = files.map(f => program.getSourceFile(join(ROOT, f)));

    // Top-level declarations: where they live and whether they are hoisted.
    const topDecl = new Map();      // declaration node -> { file, stmt, hoisted, name }
    const seen = new Map();         // name -> first file
    sources.forEach((sf, fi) => sf.statements.forEach((st, si) => {
        const add = (nameNode, declNode, hoisted, kind) => {
            const name = nameNode.text;
            topDecl.set(declNode, { file: fi, stmt: si, hoisted, name });
            if (kind !== 'var' && RESTRICTED.has(name)) problem(`${files[fi]}: top-level ${kind} '${name}' is not allowed in a browser global scope`);
            if (seen.has(name) && kind !== 'var') problem(`${contextName}: '${name}' is declared in both ${seen.get(name)} and ${files[fi]}`);
            seen.set(name, files[fi]);
        };
        if (ts.isFunctionDeclaration(st) && st.name) add(st.name, st, true, 'function');
        else if (ts.isClassDeclaration(st) && st.name) add(st.name, st, false, 'class');
        else if (ts.isVariableStatement(st)) {
            const kind = (st.declarationList.flags & ts.NodeFlags.Const) ? 'const'
                       : (st.declarationList.flags & ts.NodeFlags.Let) ? 'let' : 'var';
            const visit = (n, d) => {
                if (ts.isIdentifier(n)) add(n, d, kind === 'var', kind);
                else if (ts.isObjectBindingPattern(n) || ts.isArrayBindingPattern(n))
                    n.elements.forEach(e => e.name && visit(e.name, d));
            };
            st.declarationList.declarations.forEach(d => visit(d.name, d));
        }
    }));

    const declOf = (id) => {
        const sym = checker.getSymbolAtLocation(id);
        if (!sym || !sym.declarations) return null;
        for (const d of sym.declarations) if (topDecl.has(d)) return d;
        return null;
    };
    const isFnLike = (n) => n && (ts.isFunctionExpression(n) || ts.isArrowFunction(n) || ts.isFunctionDeclaration(n));
    const fnBodyOf = (decl) => {
        if (ts.isFunctionDeclaration(decl)) return decl;
        if (ts.isVariableDeclaration(decl) && isFnLike(decl.initializer)) return decl.initializer;
        return null;
    };

    // Walk code that runs NOW, from position (fi, si).
    const reported = new Set();
    const walkEager = (node, fi, si, stack) => {
        const visit = (n) => {
            if (!n) return;
            // Function bodies do not run where they are written ...
            if (isFnLike(n) || ts.isMethodDeclaration(n) || ts.isGetAccessor(n) || ts.isSetAccessor(n) || ts.isConstructorDeclaration(n)) return;
            if (ts.isClassDeclaration(n) || ts.isClassExpression(n)) {
                if (n.heritageClauses) n.heritageClauses.forEach(visit);
                n.members.forEach(m => {
                    if (ts.isClassStaticBlockDeclaration(m)) visit(m.body);
                    else if (ts.isPropertyDeclaration(m) && m.initializer &&
                             m.modifiers && m.modifiers.some(x => x.kind === ts.SyntaxKind.StaticKeyword)) visit(m.initializer);
                });
                return;
            }
            // ... except when called right here.
            if (ts.isCallExpression(n) || ts.isNewExpression(n)) {
                let callee = n.expression;
                while (ts.isParenthesizedExpression(callee)) callee = callee.expression;
                if (isFnLike(callee)) runFunction(callee, fi, si, stack);            // IIFE
                if (ts.isIdentifier(callee)) {
                    const d = declOf(callee);
                    if (d) {
                        check(callee, d, fi, si, stack);
                        const fn = fnBodyOf(d);
                        if (fn && ts.isCallExpression(n)) runFunction(fn, fi, si, [...stack, callee.text]);
                        if (ts.isNewExpression(n) && ts.isClassDeclaration(d)) runClass(d, fi, si, [...stack, 'new ' + callee.text]);
                    }
                } else visit(callee);
                const sync = ts.isPropertyAccessExpression(callee) && ARRAY_SYNC.has(callee.name.text);
                const promise = ts.isNewExpression(n) && ts.isIdentifier(callee) && callee.text === 'Promise';
                (n.arguments || []).forEach(a => {
                    if ((sync || promise) && isFnLike(a)) runFunction(a, fi, si, stack);
                    else visit(a);
                });
                return;
            }
            if (ts.isIdentifier(n)) {
                const p = n.parent;
                if (p && ts.isPropertyAccessExpression(p) && p.name === n) return;
                if (p && (ts.isPropertyAssignment(p) || ts.isMethodDeclaration(p)) && p.name === n) return;
                const d = declOf(n);
                if (d) check(n, d, fi, si, stack);
                return;
            }
            ts.forEachChild(n, visit);
        };
        visit(node);
    };
    const running = new Set();
    const runFunction = (fn, fi, si, stack) => {
        if (running.has(fn)) return;
        running.add(fn);
        fn.parameters.forEach(p => p.initializer && walkEager(p.initializer, fi, si, stack));
        if (fn.body) walkEager(fn.body, fi, si, stack);
        running.delete(fn);
    };
    const runClass = (cls, fi, si, stack) => {
        cls.members.forEach(m => {
            if (ts.isConstructorDeclaration(m)) runFunction(m, fi, si, stack);
            else if (ts.isPropertyDeclaration(m) && m.initializer) walkEager(m.initializer, fi, si, stack);
        });
    };
    const check = (id, decl, fi, si, stack) => {
        const info = topDecl.get(decl);
        const ok = info.file < fi || (info.file === fi && (info.hoisted || info.stmt < si));
        if (ok) return;
        const sf = sources[fi];
        const line = sf.getLineAndCharacterOfPosition(sf.statements[si].getStart(sf)).line + 1;
        const key = `${files[fi]}:${line}:${info.name}`;
        if (reported.has(key)) return;
        reported.add(key);
        const via = stack.length ? ` (via ${stack.join(' -> ')})` : '';
        problem(`${contextName}: ${files[fi]}:${line} uses '${info.name}' at load time${via}, ` +
                `but it is defined later in ${files[info.file]}`);
    };

    sources.forEach((sf, fi) => sf.statements.forEach((st, si) => {
        if (ts.isFunctionDeclaration(st)) return;
        if (ts.isClassDeclaration(st)) { walkEager(st, fi, si, []); return; }
        if (ts.isVariableStatement(st)) {
            st.declarationList.declarations.forEach(d => d.initializer && walkEager(d.initializer, fi, si, []));
            return;
        }
        walkEager(st, fi, si, []);
    }));
}

for (const [name, files] of Object.entries(contexts)) analyse(name, files);

if (problems.length) {
    console.log(`FAIL: ${problems.length} problem(s)\n`);
    for (const p of problems) console.log('  - ' + p);
    process.exit(1);
}
console.log(`PASS: ${Object.values(contexts).map(f => f.length).join(' / ')} scripts ` +
            `(main thread / CPU worker / refinement worker) load in a safe order.`);
