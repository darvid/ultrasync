"""Auto-discovery of conventions from linter configuration files.

Parses common linting tool configs (eslint, biome, ruff, etc.) and
generates conventions from enabled rules.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

LINTER_CONFIG_FILES = {
    # JavaScript/TypeScript
    "eslint": [
        ".eslintrc",
        ".eslintrc.js",
        ".eslintrc.cjs",
        ".eslintrc.json",
        ".eslintrc.yaml",
        ".eslintrc.yml",
        "eslint.config.js",
        "eslint.config.mjs",
        "eslint.config.cjs",
    ],
    "biome": ["biome.json", "biome.jsonc"],
    "oxlint": ["oxlintrc.json", ".oxlintrc.json"],
    "prettier": [".prettierrc", ".prettierrc.json", ".prettierrc.js"],
    # Python
    "ruff": ["ruff.toml", ".ruff.toml", "pyproject.toml"],
    "flake8": [".flake8", "setup.cfg", "tox.ini"],
    "isort": [".isort.cfg", "pyproject.toml", "setup.cfg"],
    "black": ["pyproject.toml"],
    "mypy": ["mypy.ini", ".mypy.ini", "pyproject.toml"],
    # Rust
    "clippy": ["clippy.toml", ".clippy.toml"],
    "rustfmt": ["rustfmt.toml", ".rustfmt.toml"],
    # Go
    "golangci-lint": [".golangci.yml", ".golangci.yaml", ".golangci.toml"],
}

# fmt: off
# ruff: noqa: E501
RULE_DESCRIPTIONS = {
    # ==========================================================================
    # ESLint Core Rules - Possible Problems
    # ==========================================================================
    "array-callback-return": "Enforce return statements in callbacks of array methods",
    "constructor-super": "Require super() calls in constructors",
    "for-direction": "Enforce for loop update clause moving counter in the right direction",
    "getter-return": "Enforce return statements in getters",
    "no-async-promise-executor": "Disallow using async function as Promise executor",
    "no-await-in-loop": "Disallow await inside of loops",
    "no-class-assign": "Disallow reassigning class members",
    "no-compare-neg-zero": "Disallow comparing against -0",
    "no-cond-assign": "Disallow assignment operators in conditional expressions",
    "no-const-assign": "Disallow reassigning const variables",
    "no-constant-binary-expression": "Disallow expressions where operation doesn't affect value",
    "no-constant-condition": "Disallow constant expressions in conditions",
    "no-constructor-return": "Disallow returning value from constructor",
    "no-control-regex": "Disallow control characters in regular expressions",
    "no-debugger": "Disallow the use of debugger",
    "no-dupe-args": "Disallow duplicate arguments in function definitions",
    "no-dupe-class-members": "Disallow duplicate class members",
    "no-dupe-else-if": "Disallow duplicate conditions in if-else-if chains",
    "no-dupe-keys": "Disallow duplicate keys in object literals",
    "no-duplicate-case": "Disallow duplicate case labels",
    "no-duplicate-imports": "Disallow duplicate module imports",
    "no-empty-character-class": "Disallow empty character classes in regular expressions",
    "no-empty-pattern": "Disallow empty destructuring patterns",
    "no-ex-assign": "Disallow reassigning exceptions in catch clauses",
    "no-fallthrough": "Disallow fallthrough of case statements",
    "no-func-assign": "Disallow reassigning function declarations",
    "no-import-assign": "Disallow assigning to imported bindings",
    "no-inner-declarations": "Disallow variable or function declarations in nested blocks",
    "no-invalid-regexp": "Disallow invalid regular expression strings in RegExp constructors",
    "no-irregular-whitespace": "Disallow irregular whitespace",
    "no-loss-of-precision": "Disallow literal numbers that lose precision",
    "no-misleading-character-class": "Disallow characters with multiple code points in char class",
    "no-new-native-nonconstructor": "Disallow new operators with global non-constructor functions",
    "no-obj-calls": "Disallow calling global object properties as functions",
    "no-promise-executor-return": "Disallow returning values from Promise executor functions",
    "no-prototype-builtins": "Disallow calling Object.prototype methods directly on objects",
    "no-self-assign": "Disallow assignments where both sides are exactly the same",
    "no-self-compare": "Disallow comparisons where both sides are exactly the same",
    "no-setter-return": "Disallow returning values from setters",
    "no-sparse-arrays": "Disallow sparse arrays",
    "no-template-curly-in-string": "Disallow template literal placeholder syntax in regular strings",
    "no-this-before-super": "Disallow this/super before calling super() in constructors",
    "no-undef": "Disallow undeclared variables",
    "no-unexpected-multiline": "Disallow confusing multiline expressions",
    "no-unmodified-loop-condition": "Disallow unmodified loop conditions",
    "no-unreachable": "Disallow unreachable code after return, throw, continue, and break",
    "no-unreachable-loop": "Disallow loops with a body that allows only one iteration",
    "no-unsafe-finally": "Disallow control flow statements in finally blocks",
    "no-unsafe-negation": "Disallow negating the left operand of relational operators",
    "no-unsafe-optional-chaining": "Disallow optional chaining where undefined is not allowed",
    "no-unused-private-class-members": "Disallow unused private class members",
    "no-unused-vars": "Disallow unused variables",
    "no-use-before-define": "Disallow using variables before they are defined",
    "no-useless-backreference": "Disallow useless backreferences in regular expressions",
    "require-atomic-updates": "Disallow assignments that can lead to race conditions",
    "use-isnan": "Require calls to isNaN() when checking for NaN",
    "valid-typeof": "Enforce comparing typeof expressions against valid strings",

    # ==========================================================================
    # ESLint Core Rules - Suggestions
    # ==========================================================================
    "accessor-pairs": "Enforce getter and setter pairs in objects and classes",
    "arrow-body-style": "Require braces around arrow function bodies",
    "block-scoped-var": "Enforce use of variables within the scope they are defined",
    "camelcase": "Enforce camelcase naming convention",
    "capitalized-comments": "Enforce capitalization of the first letter of a comment",
    "class-methods-use-this": "Enforce class methods utilize this",
    "complexity": "Enforce maximum cyclomatic complexity allowed in a program",
    "consistent-return": "Require return statements to always or never specify values",
    "consistent-this": "Enforce consistent naming when capturing execution context",
    "curly": "Enforce consistent brace style for all control statements",
    "default-case": "Require default cases in switch statements",
    "default-case-last": "Enforce default clauses in switch statements to be last",
    "default-param-last": "Enforce default parameters to be last",
    "dot-notation": "Enforce dot notation whenever possible",
    "eqeqeq": "Require use of === and !==",
    "func-name-matching": "Require function names to match variable or property names",
    "func-names": "Require or disallow named function expressions",
    "func-style": "Enforce consistent use of function declarations or expressions",
    "grouped-accessor-pairs": "Require grouped accessor pairs in object literals and classes",
    "guard-for-in": "Require for-in loops to include an if statement",
    "id-denylist": "Disallow specified identifiers",
    "id-length": "Enforce minimum and maximum identifier lengths",
    "id-match": "Require identifiers to match a specified regular expression",
    "init-declarations": "Require or disallow initialization in variable declarations",
    "logical-assignment-operators": "Require or disallow logical assignment operator shorthand",
    "max-classes-per-file": "Enforce maximum number of classes per file",
    "max-depth": "Enforce maximum depth that blocks can be nested",
    "max-lines": "Enforce maximum number of lines per file",
    "max-lines-per-function": "Enforce maximum number of lines in a function",
    "max-nested-callbacks": "Enforce maximum depth that callbacks can be nested",
    "max-params": "Enforce maximum number of parameters in function definitions",
    "max-statements": "Enforce maximum number of statements in function blocks",
    "new-cap": "Require constructor names to begin with a capital letter",
    "no-alert": "Disallow use of alert, confirm, and prompt",
    "no-array-constructor": "Disallow Array constructors",
    "no-bitwise": "Disallow bitwise operators",
    "no-caller": "Disallow use of arguments.caller or arguments.callee",
    "no-case-declarations": "Disallow lexical declarations in case clauses",
    "no-console": "Disallow use of console",
    "no-continue": "Disallow continue statements",
    "no-delete-var": "Disallow deleting variables",
    "no-div-regex": "Disallow equal signs at beginning of regular expressions",
    "no-else-return": "Disallow else blocks after return statements in if statements",
    "no-empty": "Disallow empty block statements",
    "no-empty-function": "Disallow empty functions",
    "no-empty-static-block": "Disallow empty static blocks",
    "no-eq-null": "Disallow null comparisons without type-checking operators",
    "no-eval": "Disallow use of eval()",
    "no-extend-native": "Disallow extending native types",
    "no-extra-bind": "Disallow unnecessary calls to .bind()",
    "no-extra-boolean-cast": "Disallow unnecessary boolean casts",
    "no-extra-label": "Disallow unnecessary labels",
    "no-global-assign": "Disallow assignments to native objects or read-only global variables",
    "no-implicit-coercion": "Disallow shorthand type conversions",
    "no-implicit-globals": "Disallow declarations in the global scope",
    "no-implied-eval": "Disallow use of eval()-like methods",
    "no-inline-comments": "Disallow inline comments after code",
    "no-invalid-this": "Disallow this keywords outside of classes or class-like objects",
    "no-iterator": "Disallow use of the __iterator__ property",
    "no-label-var": "Disallow labels that share a name with a variable",
    "no-labels": "Disallow labeled statements",
    "no-lone-blocks": "Disallow unnecessary nested blocks",
    "no-lonely-if": "Disallow if statements as the only statement in else blocks",
    "no-loop-func": "Disallow function declarations with unsafe references inside loops",
    "no-magic-numbers": "Disallow magic numbers",
    "no-multi-assign": "Disallow use of chained assignment expressions",
    "no-multi-str": "Disallow multiline strings",
    "no-negated-condition": "Disallow negated conditions",
    "no-nested-ternary": "Disallow nested ternary expressions",
    "no-new": "Disallow new operators outside of assignments or comparisons",
    "no-new-func": "Disallow new operators with the Function object",
    "no-new-wrappers": "Disallow new operators with String, Number, and Boolean objects",
    "no-nonoctal-decimal-escape": "Disallow \\8 and \\9 escape sequences in string literals",
    "no-object-constructor": "Disallow calls to Object constructor without an argument",
    "no-octal": "Disallow octal literals",
    "no-octal-escape": "Disallow octal escape sequences in string literals",
    "no-param-reassign": "Disallow reassigning function parameters",
    "no-plusplus": "Disallow unary operators ++ and --",
    "no-proto": "Disallow use of the __proto__ property",
    "no-redeclare": "Disallow variable redeclaration",
    "no-regex-spaces": "Disallow multiple spaces in regular expressions",
    "no-restricted-exports": "Disallow specified names in exports",
    "no-restricted-globals": "Disallow specified global variables",
    "no-restricted-imports": "Disallow specified modules when loaded by import",
    "no-restricted-properties": "Disallow certain properties on certain objects",
    "no-restricted-syntax": "Disallow specified syntax",
    "no-return-assign": "Disallow assignment operators in return statements",
    "no-return-await": "Disallow unnecessary return await",
    "no-script-url": "Disallow javascript: URLs",
    "no-sequences": "Disallow comma operators",
    "no-shadow": "Disallow variable declarations from shadowing outer scope variables",
    "no-shadow-restricted-names": "Disallow identifiers from shadowing restricted names",
    "no-ternary": "Disallow ternary operators",
    "no-throw-literal": "Disallow throwing literals as exceptions",
    "no-undef-init": "Disallow initializing variables to undefined",
    "no-undefined": "Disallow use of undefined as an identifier",
    "no-underscore-dangle": "Disallow dangling underscores in identifiers",
    "no-unneeded-ternary": "Disallow ternary operators when simpler alternatives exist",
    "no-unused-expressions": "Disallow unused expressions",
    "no-unused-labels": "Disallow unused labels",
    "no-useless-call": "Disallow unnecessary calls to .call() and .apply()",
    "no-useless-catch": "Disallow unnecessary catch clauses",
    "no-useless-computed-key": "Disallow unnecessary computed property keys in objects/classes",
    "no-useless-concat": "Disallow unnecessary concatenation of literals or template literals",
    "no-useless-constructor": "Disallow unnecessary constructors",
    "no-useless-escape": "Disallow unnecessary escape characters",
    "no-useless-rename": "Disallow renaming import, export, destructured assignments to same name",
    "no-useless-return": "Disallow redundant return statements",
    "no-var": "Require let or const instead of var",
    "no-void": "Disallow void operators",
    "no-warning-comments": "Disallow specified warning terms in comments",
    "no-with": "Disallow with statements",
    "object-shorthand": "Require or disallow method and property shorthand syntax",
    "one-var": "Enforce variables to be declared together or separately in functions",
    "operator-assignment": "Require or disallow assignment operator shorthand where possible",
    "prefer-arrow-callback": "Require using arrow functions for callbacks",
    "prefer-const": "Require const declarations for variables never reassigned",
    "prefer-destructuring": "Require destructuring from arrays and/or objects",
    "prefer-exponentiation-operator": "Disallow use of Math.pow in favor of ** operator",
    "prefer-named-capture-group": "Enforce using named capture group in regular expression",
    "prefer-numeric-literals": "Disallow parseInt() in favor of binary, octal, and hex literals",
    "prefer-object-has-own": "Disallow Object.prototype.hasOwnProperty.call() for Object.hasOwn()",
    "prefer-object-spread": "Disallow Object.assign with object literal as first argument",
    "prefer-promise-reject-errors": "Require Error objects as Promise rejection reasons",
    "prefer-regex-literals": "Disallow use of RegExp constructor in favor of regex literals",
    "prefer-rest-params": "Require rest parameters instead of arguments",
    "prefer-spread": "Require spread operators instead of .apply()",
    "prefer-template": "Require template literals instead of string concatenation",
    "radix": "Enforce consistent use of radix argument when using parseInt()",
    "require-await": "Disallow async functions which have no await expression",
    "require-unicode-regexp": "Enforce use of u or v flag on regular expressions",
    "require-yield": "Require generator functions to contain yield",
    "sort-imports": "Enforce sorted import declarations within modules",
    "sort-keys": "Require object keys to be sorted",
    "sort-vars": "Require variables within same declaration block to be sorted",
    "strict": "Require or disallow strict mode directives",
    "symbol-description": "Require symbol descriptions",
    "vars-on-top": "Require var declarations be placed at top of their containing scope",
    "yoda": "Require or disallow Yoda conditions",

    # Formatting
    "semi": "Require or disallow semicolons instead of ASI",
    "quotes": "Enforce consistent use of backticks, double, or single quotes",
    "indent": "Enforce consistent indentation",
    "max-len": "Enforce a maximum line length",
    "no-trailing-spaces": "Disallow trailing whitespace at end of lines",
    "comma-dangle": "Require or disallow trailing commas",
    "unicode-bom": "Require or disallow Unicode byte order mark (BOM)",

    # ==========================================================================
    # TypeScript-ESLint Rules
    # ==========================================================================
    "@typescript-eslint/adjacent-overload-signatures": "Require function overload signatures be consecutive",
    "@typescript-eslint/array-type": "Require consistently using either T[] or Array<T> for arrays",
    "@typescript-eslint/await-thenable": "Disallow awaiting a value that is not a Thenable",
    "@typescript-eslint/ban-ts-comment": "Disallow @ts-<directive> comments or require descriptions",
    "@typescript-eslint/ban-tslint-comment": "Disallow // tslint:<rule-flag> comments",
    "@typescript-eslint/class-literal-property-style": "Enforce literals on classes are exposed consistently",
    "@typescript-eslint/class-methods-use-this": "Enforce class methods utilize this",
    "@typescript-eslint/consistent-generic-constructors": "Enforce specifying generic type arguments on type annotation or constructor",
    "@typescript-eslint/consistent-indexed-object-style": "Require or disallow the Record type",
    "@typescript-eslint/consistent-return": "Require return statements to always or never specify values",
    "@typescript-eslint/consistent-type-assertions": "Enforce consistent usage of type assertions",
    "@typescript-eslint/consistent-type-definitions": "Enforce type definitions use either interface or type",
    "@typescript-eslint/consistent-type-exports": "Enforce consistent usage of type exports",
    "@typescript-eslint/consistent-type-imports": "Enforce consistent usage of type imports",
    "@typescript-eslint/default-param-last": "Enforce default parameters to be last",
    "@typescript-eslint/dot-notation": "Enforce dot notation whenever possible",
    "@typescript-eslint/explicit-function-return-type": "Require explicit return types on functions and class methods",
    "@typescript-eslint/explicit-member-accessibility": "Require explicit accessibility modifiers on class properties and methods",
    "@typescript-eslint/explicit-module-boundary-types": "Require explicit return and argument types on exported functions",
    "@typescript-eslint/init-declarations": "Require or disallow initialization in variable declarations",
    "@typescript-eslint/max-params": "Enforce maximum number of parameters in function definitions",
    "@typescript-eslint/member-ordering": "Require consistent member declaration order",
    "@typescript-eslint/method-signature-style": "Enforce using a particular method signature syntax",
    "@typescript-eslint/naming-convention": "Enforce naming conventions for everything across a codebase",
    "@typescript-eslint/no-array-constructor": "Disallow generic Array constructors",
    "@typescript-eslint/no-array-delete": "Disallow using the delete operator on array values",
    "@typescript-eslint/no-base-to-string": "Require .toString() only on objects providing useful information",
    "@typescript-eslint/no-confusing-non-null-assertion": "Disallow non-null assertion in confusing locations",
    "@typescript-eslint/no-confusing-void-expression": "Require expressions of type void appear in statement position",
    "@typescript-eslint/no-deprecated": "Disallow using code marked as @deprecated",
    "@typescript-eslint/no-dupe-class-members": "Disallow duplicate class members",
    "@typescript-eslint/no-duplicate-enum-values": "Disallow duplicate enum member values",
    "@typescript-eslint/no-duplicate-type-constituents": "Disallow duplicate constituents of union or intersection types",
    "@typescript-eslint/no-dynamic-delete": "Disallow using the delete operator on computed key expressions",
    "@typescript-eslint/no-empty-function": "Disallow empty functions",
    "@typescript-eslint/no-empty-interface": "Disallow declaration of empty interfaces",
    "@typescript-eslint/no-empty-object-type": "Disallow accidentally using the empty object type",
    "@typescript-eslint/no-explicit-any": "Disallow the any type",
    "@typescript-eslint/no-extra-non-null-assertion": "Disallow extra non-null assertions",
    "@typescript-eslint/no-extraneous-class": "Disallow classes used as namespaces",
    "@typescript-eslint/no-floating-promises": "Require Promise-like statements be handled appropriately",
    "@typescript-eslint/no-for-in-array": "Disallow iterating over an array with a for-in loop",
    "@typescript-eslint/no-implied-eval": "Disallow use of eval()-like functions",
    "@typescript-eslint/no-import-type-side-effects": "Enforce use of top-level import type qualifier",
    "@typescript-eslint/no-inferrable-types": "Disallow explicit type declarations for initialized primitives",
    "@typescript-eslint/no-invalid-this": "Disallow this keywords outside of classes or class-like objects",
    "@typescript-eslint/no-invalid-void-type": "Disallow void type outside of generic or return types",
    "@typescript-eslint/no-loop-func": "Disallow function declarations with unsafe references inside loops",
    "@typescript-eslint/no-loss-of-precision": "Disallow literal numbers that lose precision",
    "@typescript-eslint/no-magic-numbers": "Disallow magic numbers",
    "@typescript-eslint/no-meaningless-void-operator": "Disallow void operator except when used to discard a value",
    "@typescript-eslint/no-misused-new": "Enforce valid definition of new and constructor",
    "@typescript-eslint/no-misused-promises": "Disallow Promises in places not designed to handle them",
    "@typescript-eslint/no-misused-spread": "Disallow using spread operator when it might cause unexpected behavior",
    "@typescript-eslint/no-mixed-enums": "Disallow enums from having both number and string members",
    "@typescript-eslint/no-namespace": "Disallow TypeScript namespaces",
    "@typescript-eslint/no-non-null-asserted-nullish-coalescing": "Disallow non-null assertions in nullish coalescing left operand",
    "@typescript-eslint/no-non-null-asserted-optional-chain": "Disallow non-null assertions after optional chain expression",
    "@typescript-eslint/no-non-null-assertion": "Disallow non-null assertions using the ! postfix operator",
    "@typescript-eslint/no-redeclare": "Disallow variable redeclaration",
    "@typescript-eslint/no-redundant-type-constituents": "Disallow members of unions and intersections that do nothing",
    "@typescript-eslint/no-require-imports": "Disallow invocation of require()",
    "@typescript-eslint/no-restricted-imports": "Disallow specified modules when loaded by import",
    "@typescript-eslint/no-restricted-types": "Disallow certain types",
    "@typescript-eslint/no-shadow": "Disallow variable declarations from shadowing outer scope variables",
    "@typescript-eslint/no-this-alias": "Disallow aliasing this",
    "@typescript-eslint/no-unnecessary-boolean-literal-compare": "Disallow unnecessary equality comparisons against boolean literals",
    "@typescript-eslint/no-unnecessary-condition": "Disallow conditionals where type is always truthy or falsy",
    "@typescript-eslint/no-unnecessary-parameter-property-assignment": "Disallow unnecessary assignment of constructor property parameter",
    "@typescript-eslint/no-unnecessary-qualifier": "Disallow unnecessary namespace qualifiers",
    "@typescript-eslint/no-unnecessary-template-expression": "Disallow unnecessary template expressions",
    "@typescript-eslint/no-unnecessary-type-arguments": "Disallow type arguments equal to the default",
    "@typescript-eslint/no-unnecessary-type-assertion": "Disallow type assertions that do not change expression type",
    "@typescript-eslint/no-unnecessary-type-constraint": "Disallow unnecessary constraints on generic types",
    "@typescript-eslint/no-unnecessary-type-conversion": "Disallow conversion idioms when they do not change type or value",
    "@typescript-eslint/no-unnecessary-type-parameters": "Disallow type parameters that aren't used multiple times",
    "@typescript-eslint/no-unsafe-argument": "Disallow calling a function with a value with type any",
    "@typescript-eslint/no-unsafe-assignment": "Disallow assigning a value with type any to variables and properties",
    "@typescript-eslint/no-unsafe-call": "Disallow calling a value with type any",
    "@typescript-eslint/no-unsafe-declaration-merging": "Disallow unsafe declaration merging",
    "@typescript-eslint/no-unsafe-enum-comparison": "Disallow comparing an enum value with a non-enum value",
    "@typescript-eslint/no-unsafe-function-type": "Disallow using the unsafe built-in Function type",
    "@typescript-eslint/no-unsafe-member-access": "Disallow member access on a value with type any",
    "@typescript-eslint/no-unsafe-return": "Disallow returning a value with type any from a function",
    "@typescript-eslint/no-unsafe-type-assertion": "Disallow type assertions that narrow a type",
    "@typescript-eslint/no-unsafe-unary-minus": "Require unary negation to take a number",
    "@typescript-eslint/no-unused-expressions": "Disallow unused expressions",
    "@typescript-eslint/no-unused-vars": "Disallow unused variables",
    "@typescript-eslint/no-use-before-define": "Disallow using variables before they are defined",
    "@typescript-eslint/no-useless-constructor": "Disallow unnecessary constructors",
    "@typescript-eslint/no-useless-empty-export": "Disallow empty exports that don't change anything in a module file",
    "@typescript-eslint/no-var-requires": "Disallow require statements except in import statements",
    "@typescript-eslint/no-wrapper-object-types": "Disallow using confusing built-in primitive class wrappers",
    "@typescript-eslint/non-nullable-type-assertion-style": "Enforce non-null assertions over explicit type assertions",
    "@typescript-eslint/only-throw-error": "Disallow throwing non-Error values as exceptions",
    "@typescript-eslint/parameter-properties": "Require or disallow parameter properties in class constructors",
    "@typescript-eslint/prefer-as-const": "Enforce use of as const over literal type",
    "@typescript-eslint/prefer-destructuring": "Require destructuring from arrays and/or objects",
    "@typescript-eslint/prefer-enum-initializers": "Require each enum member value be explicitly initialized",
    "@typescript-eslint/prefer-find": "Enforce Array.prototype.find() over filter() followed by [0]",
    "@typescript-eslint/prefer-for-of": "Enforce use of for-of loop over the standard for loop",
    "@typescript-eslint/prefer-function-type": "Enforce using function types instead of interfaces with call signatures",
    "@typescript-eslint/prefer-includes": "Enforce includes method over indexOf method",
    "@typescript-eslint/prefer-literal-enum-member": "Require all enum members to be literal values",
    "@typescript-eslint/prefer-namespace-keyword": "Require using namespace keyword over module keyword",
    "@typescript-eslint/prefer-nullish-coalescing": "Enforce using nullish coalescing operator instead of logical assignments",
    "@typescript-eslint/prefer-optional-chain": "Enforce using concise optional chain expressions instead of chained logical ands",
    "@typescript-eslint/prefer-promise-reject-errors": "Require using Error objects as Promise rejection reasons",
    "@typescript-eslint/prefer-readonly": "Require private members marked as readonly if never modified outside constructor",
    "@typescript-eslint/prefer-readonly-parameter-types": "Require function parameters typed as readonly to prevent mutation",
    "@typescript-eslint/prefer-reduce-type-parameter": "Enforce using type parameter when calling Array#reduce",
    "@typescript-eslint/prefer-regexp-exec": "Enforce RegExp#exec over String#match if no global flag provided",
    "@typescript-eslint/prefer-return-this-type": "Enforce that this is used when only this type is returned",
    "@typescript-eslint/prefer-string-starts-ends-with": "Enforce String#startsWith and String#endsWith over other methods",
    "@typescript-eslint/prefer-ts-expect-error": "Enforce using @ts-expect-error over @ts-ignore",
    "@typescript-eslint/promise-function-async": "Require any function returning a Promise be marked async",
    "@typescript-eslint/require-array-sort-compare": "Require Array#sort and Array#toSorted calls provide a compareFunction",
    "@typescript-eslint/require-await": "Disallow async functions which have no await expression",
    "@typescript-eslint/restrict-plus-operands": "Require both operands of addition be the same type",
    "@typescript-eslint/restrict-template-expressions": "Enforce template literal expressions be of string type",
    "@typescript-eslint/return-await": "Enforce consistent awaiting of returned promises",
    "@typescript-eslint/sort-type-constituents": "Enforce constituents of type union/intersection be sorted alphabetically",
    "@typescript-eslint/strict-boolean-expressions": "Disallow certain types in boolean expressions",
    "@typescript-eslint/switch-exhaustiveness-check": "Require switch-case statements be exhaustive",
    "@typescript-eslint/triple-slash-reference": "Disallow certain triple slash directives in favor of ES6-style imports",
    "@typescript-eslint/typedef": "Require type annotations in certain places",
    "@typescript-eslint/unbound-method": "Enforce unbound methods are called with their expected scope",
    "@typescript-eslint/unified-signatures": "Disallow two overloads that could be unified into one",
    "@typescript-eslint/use-unknown-in-catch-callback-variable": "Enforce typing arguments in Promise rejection callbacks as unknown",

    # ==========================================================================
    # React ESLint Plugin Rules
    # ==========================================================================
    "react/boolean-prop-naming": "Enforce consistent naming for boolean props",
    "react/button-has-type": "Disallow button elements without an explicit type attribute",
    "react/checked-requires-onchange-or-readonly": "Enforce onChange or readonly when checked is used",
    "react/default-props-match-prop-types": "Enforce defaultProps have a corresponding non-required PropType",
    "react/destructuring-assignment": "Enforce consistent usage of destructuring assignment of props, state, context",
    "react/display-name": "Disallow missing displayName in a React component definition",
    "react/forbid-component-props": "Disallow certain props on components",
    "react/forbid-dom-props": "Disallow certain props on DOM Nodes",
    "react/forbid-elements": "Disallow certain elements",
    "react/forbid-foreign-prop-types": "Disallow using another component's propTypes",
    "react/forbid-prop-types": "Disallow certain propTypes",
    "react/forward-ref-uses-ref": "Require all forwardRef components include a ref parameter",
    "react/function-component-definition": "Enforce a specific function type for function components",
    "react/hook-use-state": "Ensure symmetric naming of useState hook value and setter variables",
    "react/iframe-missing-sandbox": "Enforce sandbox attribute on iframe elements",
    "react/jsx-boolean-value": "Enforce boolean attributes notation in JSX",
    "react/jsx-child-element-spacing": "Enforce spacing in curly braces in JSX attributes and expressions",
    "react/jsx-closing-bracket-location": "Enforce closing bracket location in JSX",
    "react/jsx-closing-tag-location": "Enforce closing tag location for multiline JSX",
    "react/jsx-curly-brace-presence": "Disallow unnecessary JSX expressions when literals alone are sufficient",
    "react/jsx-curly-newline": "Enforce consistent linebreaks in curly braces in JSX",
    "react/jsx-curly-spacing": "Enforce spacing inside of curly braces in JSX attributes and expressions",
    "react/jsx-equals-spacing": "Enforce spacing around equal signs in JSX attributes",
    "react/jsx-filename-extension": "Disallow file extensions that may contain JSX",
    "react/jsx-first-prop-new-line": "Enforce proper position of the first property in JSX",
    "react/jsx-fragments": "Enforce shorthand or standard form for React fragments",
    "react/jsx-handler-names": "Enforce event handler naming conventions in JSX",
    "react/jsx-indent": "Enforce JSX indentation",
    "react/jsx-indent-props": "Enforce props indentation in JSX",
    "react/jsx-key": "Disallow missing key props in iterators/collection literals",
    "react/jsx-max-depth": "Enforce JSX maximum depth",
    "react/jsx-max-props-per-line": "Enforce maximum of props on a single line in JSX",
    "react/jsx-newline": "Require or prevent a new line after jsx elements and expressions",
    "react/jsx-no-bind": "Disallow .bind() or arrow functions in JSX props",
    "react/jsx-no-comment-textnodes": "Disallow comments from being inserted as text nodes",
    "react/jsx-no-constructed-context-values": "Disallow passing constructed values to context providers",
    "react/jsx-no-duplicate-props": "Disallow duplicate properties in JSX",
    "react/jsx-no-leaked-render": "Disallow problematic leaked values from being rendered",
    "react/jsx-no-literals": "Disallow usage of string literals in JSX",
    "react/jsx-no-script-url": "Disallow usage of javascript: URLs",
    "react/jsx-no-target-blank": "Disallow target='_blank' attribute without rel='noreferrer'",
    "react/jsx-no-undef": "Disallow undeclared variables in JSX",
    "react/jsx-no-useless-fragment": "Disallow unnecessary fragments",
    "react/jsx-one-expression-per-line": "Require one JSX element per line",
    "react/jsx-pascal-case": "Enforce PascalCase for user-defined JSX components",
    "react/jsx-props-no-multi-spaces": "Disallow multiple spaces between inline JSX props",
    "react/jsx-props-no-spreading": "Disallow JSX prop spreading",
    "react/jsx-sort-props": "Enforce props alphabetical sorting",
    "react/jsx-tag-spacing": "Enforce whitespace in and around the JSX opening and closing brackets",
    "react/jsx-uses-react": "Disallow React being incorrectly marked as unused",
    "react/jsx-uses-vars": "Disallow variables used in JSX being incorrectly marked as unused",
    "react/jsx-wrap-multilines": "Disallow missing parentheses around multiline JSX",
    "react/no-access-state-in-setstate": "Disallow using this.state within setState",
    "react/no-adjacent-inline-elements": "Disallow adjacent inline elements not separated by whitespace",
    "react/no-array-index-key": "Disallow using array index as key",
    "react/no-arrow-function-lifecycle": "Disallow arrow function lifecycle methods",
    "react/no-children-prop": "Disallow passing children as props",
    "react/no-danger": "Disallow usage of dangerous JSX properties",
    "react/no-danger-with-children": "Disallow when using dangerouslySetInnerHTML and children",
    "react/no-deprecated": "Disallow usage of deprecated methods",
    "react/no-did-mount-set-state": "Disallow usage of setState in componentDidMount",
    "react/no-did-update-set-state": "Disallow usage of setState in componentDidUpdate",
    "react/no-direct-mutation-state": "Disallow direct mutation of this.state",
    "react/no-find-dom-node": "Disallow usage of findDOMNode",
    "react/no-invalid-html-attribute": "Disallow usage of invalid HTML attributes",
    "react/no-is-mounted": "Disallow usage of isMounted",
    "react/no-multi-comp": "Disallow multiple component definition per file",
    "react/no-namespace": "Disallow usage of namespace in component name",
    "react/no-object-type-as-default-prop": "Disallow usage of object type as default prop",
    "react/no-redundant-should-component-update": "Disallow shouldComponentUpdate when extending React.PureComponent",
    "react/no-render-return-value": "Disallow usage of the return value of ReactDOM.render",
    "react/no-set-state": "Disallow usage of setState",
    "react/no-string-refs": "Disallow using string references",
    "react/no-this-in-sfc": "Disallow this from being used in stateless functional components",
    "react/no-typos": "Disallow common typos",
    "react/no-unescaped-entities": "Disallow unescaped HTML entities from appearing in markup",
    "react/no-unknown-property": "Disallow usage of unknown DOM property",
    "react/no-unsafe": "Disallow usage of unsafe lifecycle methods",
    "react/no-unstable-nested-components": "Disallow creating unstable components inside components",
    "react/no-unused-class-component-methods": "Disallow declaring unused methods of component class",
    "react/no-unused-prop-types": "Disallow definitions of unused propTypes",
    "react/no-unused-state": "Disallow definitions of unused state",
    "react/no-will-update-set-state": "Disallow usage of setState in componentWillUpdate",
    "react/prefer-es6-class": "Enforce ES6 class for defining components",
    "react/prefer-exact-props": "Enforce exact proptypes definitions",
    "react/prefer-read-only-props": "Enforce read-only props with Flow readonly modifier",
    "react/prefer-stateless-function": "Enforce stateless components to be written as pure functions",
    "react/prop-types": "Disallow missing props validation in a React component definition",
    "react/react-in-jsx-scope": "Disallow missing React when using JSX",
    "react/require-default-props": "Enforce a defaultProps definition for every prop that is not required",
    "react/require-optimization": "Enforce components to define shouldComponentUpdate method",
    "react/require-render-return": "Enforce ES5 or ES6 class for returning value in render function",
    "react/self-closing-comp": "Disallow extra closing tags for components without children",
    "react/sort-comp": "Enforce component methods order",
    "react/sort-default-props": "Enforce defaultProps declarations alphabetical sorting",
    "react/sort-prop-types": "Enforce propTypes declarations alphabetical sorting",
    "react/state-in-constructor": "Enforce state initialization style",
    "react/static-property-placement": "Enforce where React component static properties should be positioned",
    "react/style-prop-object": "Enforce style prop value is an object",
    "react/void-dom-elements-no-children": "Disallow void DOM elements from receiving children",

    # ==========================================================================
    # React Hooks ESLint Plugin Rules
    # ==========================================================================
    "react-hooks/rules-of-hooks": "Enforce fundamental principles for using React hooks correctly",
    "react-hooks/exhaustive-deps": "Ensure all dependencies are properly included in hook dependency arrays",

    # ==========================================================================
    # Ruff/Python Rules - Prefixes (Category-level)
    # ==========================================================================
    "AIR": "Airflow - Airflow-specific linting rules",
    "ANN": "flake8-annotations - Type annotation requirements",
    "ASYNC": "flake8-async - Async/await best practices",
    "ARG": "flake8-unused-arguments - Unused parameter detection",
    "B": "flake8-bugbear - Common bugs and design problems",
    "BLE": "flake8-blind-except - Exception handling anti-patterns",
    "C4": "flake8-comprehensions - Comprehension improvements",
    "C90": "mccabe - Cyclomatic complexity checking",
    "COM": "flake8-commas - Trailing comma conventions",
    "D": "pydocstyle - Docstring conventions",
    "DOC": "pydoclint - Docstring parameter documentation",
    "DJ": "flake8-django - Django framework linting",
    "DTZ": "flake8-datetimez - Timezone-aware datetime usage",
    "E": "pycodestyle errors - PEP 8 compliance issues",
    "EM": "flake8-errmsg - Exception message formatting",
    "ERA": "eradicate - Commented code detection",
    "EXE": "flake8-executable - Shebang and executable checks",
    "F": "Pyflakes - Logical errors and undefined names",
    "FAST": "FastAPI - FastAPI-specific patterns",
    "FBT": "flake8-boolean-trap - Boolean parameter anti-patterns",
    "FIX": "flake8-fixme - TODO/FIXME/HACK markers",
    "FLY": "flynt - String formatting improvements",
    "G": "flake8-logging-format - Logging statement formatting",
    "I": "isort - Import sorting",
    "ICN": "flake8-import-conventions - Import alias conventions",
    "INP": "flake8-no-pep420 - Namespace package detection",
    "INT": "flake8-gettext - Internationalization patterns",
    "ISC": "flake8-implicit-str-concat - String concatenation",
    "LOG": "flake8-logging - Logging best practices",
    "N": "pep8-naming - Naming convention violations",
    "NPY": "NumPy-specific - NumPy deprecations and best practices",
    "PD": "pandas-vet - Pandas anti-patterns",
    "PERF": "Perflint - Performance optimization suggestions",
    "PIE": "flake8-pie - Miscellaneous improvements",
    "PL": "Pylint - Pylint-style checks",
    "PLC": "Pylint Convention - Pylint convention checks",
    "PLE": "Pylint Error - Pylint error checks",
    "PLR": "Pylint Refactor - Pylint refactoring suggestions",
    "PLW": "Pylint Warning - Pylint warning checks",
    "PT": "flake8-pytest-style - pytest conventions",
    "PTH": "flake8-use-pathlib - pathlib usage",
    "PYI": "flake8-pyi - Stub file conventions",
    "Q": "flake8-quotes - Quote style consistency",
    "RET": "flake8-return - Return statement patterns",
    "RSE": "flake8-raise - Exception raising patterns",
    "RUF": "Ruff-specific - Rules unique to Ruff",
    "S": "flake8-bandit - Security issues",
    "SIM": "flake8-simplify - Code simplification",
    "SLF": "flake8-self - Private member access",
    "SLOT": "flake8-slots - __slots__ recommendations",
    "T10": "flake8-debugger - Debugger detection",
    "T20": "flake8-print - Print statement detection",
    "TC": "flake8-type-checking - Type-checking block management",
    "TD": "flake8-todos - TODO comment format",
    "TID": "flake8-tidy-imports - Import tidiness",
    "TRY": "tryceratops - Exception handling patterns",
    "UP": "pyupgrade - Python syntax modernization",
    "W": "pycodestyle warnings - PEP 8 style warnings",
    "YTT": "flake8-2020 - Python 2/3 version checks",

    # ==========================================================================
    # Ruff/Python Rules - Individual Rules
    # ==========================================================================
    # Airflow
    "AIR001": "Task variable name should match the task_id",
    "AIR002": "DAG missing schedule argument",

    # flake8-annotations
    "ANN001": "Missing function argument type annotation",
    "ANN002": "Missing type annotation for *args",
    "ANN003": "Missing type annotation for **kwargs",
    "ANN101": "Missing type annotation for self in method",
    "ANN102": "Missing type annotation for cls in classmethod",
    "ANN201": "Missing return type annotation for public function",
    "ANN202": "Missing return type annotation for private function",
    "ANN204": "Missing return type annotation for special method",
    "ANN205": "Missing return type annotation for staticmethod",
    "ANN206": "Missing return type annotation for classmethod",
    "ANN401": "Dynamically typed expressions (typing.Any) are disallowed",

    # flake8-async
    "ASYNC100": "Trio/asyncio context missing await statements",
    "ASYNC210": "Blocking HTTP call in async function",
    "ASYNC220": "Blocking HTTP call in async context manager",

    # flake8-unused-arguments
    "ARG001": "Unused function argument",
    "ARG002": "Unused method argument",
    "ARG003": "Unused class method argument",
    "ARG004": "Unused static method argument",
    "ARG005": "Unused lambda argument",

    # flake8-bugbear
    "B002": "Python does not support unary prefix increment",
    "B003": "Assignment to os.environ does not clear the environment",
    "B004": "Using hasattr(x, '__call__') instead of callable(x)",
    "B005": "Using .strip() with multi-character strings is misleading",
    "B006": "Do not use mutable data structures for argument defaults",
    "B007": "Loop control variable not used within loop body",
    "B008": "Do not perform function call in argument defaults",
    "B009": "Do not call getattr with a constant attribute value",
    "B010": "Do not call setattr with a constant attribute value",
    "B011": "Do not call assert False, raise AssertionError instead",
    "B012": "Return statement in finally block may swallow exceptions",
    "B013": "Redundant tuple in exception handler",
    "B014": "Exception handler with duplicate exceptions",
    "B015": "Pointless comparison (==, !=, is, is not)",
    "B016": "Cannot raise a literal",
    "B017": "assertRaises(Exception) or pytest.raises(Exception) is too broad",
    "B018": "Useless expression - has no side effect",
    "B019": "Use of functools.lru_cache or functools.cache on methods",
    "B020": "Loop control variable overrides iterable",
    "B021": "f-string used as docstring",
    "B022": "No arguments passed to contextlib.suppress",
    "B023": "Function definition does not bind loop variable",
    "B024": "Abstract base class has no abstract methods",
    "B025": "Exception handler with duplicate exceptions via inheritance",
    "B026": "Star-arg unpacking after a keyword argument is strong anti-pattern",
    "B027": "Empty method in abstract base class without abstract decorator",
    "B028": "No explicit stacklevel keyword argument found in warnings.warn",
    "B029": "Using except(): with an empty tuple is the same as except Exception:",
    "B030": "Except handlers should only be exception classes or tuples",
    "B031": "Using groupby() iterable in for-loop without a list conversion",
    "B032": "Possible unintentional type annotation",
    "B033": "Sets should not contain duplicate items",
    "B034": "re.sub/re.subn/re.split called with pos and endpos params",
    "B035": "Dictionary comprehension uses static key",
    "B904": "Within an except clause, raise exceptions with 'from err' or 'from None'",
    "B905": "zip() without an explicit strict= parameter",

    # flake8-blind-except
    "BLE001": "Do not catch blind exception: Exception",

    # flake8-comprehensions
    "C400": "Unnecessary generator - rewrite as a list comprehension",
    "C401": "Unnecessary generator - rewrite as a set comprehension",
    "C402": "Unnecessary generator - rewrite as a dict comprehension",
    "C403": "Unnecessary list comprehension - rewrite as a set comprehension",
    "C404": "Unnecessary list comprehension - rewrite as a dict comprehension",
    "C405": "Unnecessary literal - rewrite as a set literal",
    "C406": "Unnecessary literal - rewrite as a dict literal",
    "C408": "Unnecessary dict call - rewrite as a literal",
    "C409": "Unnecessary list passed to tuple() - rewrite as a tuple literal",
    "C410": "Unnecessary list passed to list() - use a list literal",
    "C411": "Unnecessary list call - remove the outer call to list()",
    "C413": "Unnecessary list/reversed call around sorted()",
    "C414": "Unnecessary list/reversed/set/sorted/tuple call within list/set/sorted/tuple",
    "C415": "Unnecessary subscript reversal of iterable within reversed/sorted",
    "C416": "Unnecessary list/set comprehension - rewrite using list/set()",
    "C417": "Unnecessary use of map - rewrite using a generator expression",
    "C418": "Unnecessary dict passed to dict() or dict comprehension",
    "C419": "Unnecessary list comprehension passed to any()/all()",

    # mccabe
    "C901": "Function is too complex",

    # flake8-commas
    "COM812": "Trailing comma missing",
    "COM818": "Trailing comma on bare tuple prohibited",
    "COM819": "Trailing comma prohibited",

    # pydocstyle
    "D100": "Missing docstring in public module",
    "D101": "Missing docstring in public class",
    "D102": "Missing docstring in public method",
    "D103": "Missing docstring in public function",
    "D104": "Missing docstring in public package",
    "D105": "Missing docstring in magic method",
    "D106": "Missing docstring in public nested class",
    "D107": "Missing docstring in __init__",
    "D200": "One-line docstring should fit on one line",
    "D201": "No blank lines allowed before function docstring",
    "D202": "No blank lines allowed after function docstring",
    "D203": "1 blank line required before class docstring",
    "D204": "1 blank line required after class docstring",
    "D205": "1 blank line required between summary line and description",
    "D206": "Docstring should be indented with spaces, not tabs",
    "D207": "Docstring is under-indented",
    "D208": "Docstring is over-indented",
    "D209": "Multi-line docstring closing quotes should be on separate line",
    "D210": "No whitespace allowed surrounding docstring text",
    "D211": "No blank lines allowed before class docstring",
    "D212": "Multi-line docstring summary should start at the first line",
    "D213": "Multi-line docstring summary should start at the second line",
    "D214": "Section is over-indented",
    "D215": "Section underline is over-indented",
    "D300": "Use triple double quotes",
    "D301": "Use r if any backslashes in a docstring",
    "D400": "First line should end with a period",
    "D401": "First line of docstring should be in imperative mood",
    "D402": "First line should not be the function's signature",
    "D403": "First word of the first line should be capitalized",
    "D404": "First word of the docstring should not be This",
    "D405": "Section name should be properly capitalized",
    "D406": "Section name should end with a newline",
    "D407": "Missing dashed underline after section",
    "D408": "Section underline should be in the line following the section name",
    "D409": "Section underline should match the length of its name",
    "D410": "Missing blank line after section",
    "D411": "Missing blank line before section",
    "D412": "No blank lines allowed between section header and its content",
    "D413": "Missing blank line after last section",
    "D414": "Section has no content",
    "D415": "First line should end with a period, question mark, or exclamation point",
    "D416": "Section name should end with a colon",
    "D417": "Missing argument descriptions in the docstring",
    "D418": "Function decorated with @overload shouldn't contain a docstring",
    "D419": "Docstring is empty",

    # flake8-django
    "DJ001": "Avoid using null=True on string-based fields",
    "DJ003": "Avoid passing locals() as context to a render function",
    "DJ006": "Do not use exclude with ModelForm, use fields instead",
    "DJ007": "Do not use __all__ with ModelForm, use fields instead",
    "DJ008": "Model does not define __str__ method",
    "DJ012": "Order of model's inner classes, methods, and fields does not follow the Django Style Guide",

    # flake8-datetimez
    "DTZ001": "datetime.datetime() called without a tzinfo argument",
    "DTZ002": "datetime.datetime.today() used - prefer datetime.datetime.now(tz=)",
    "DTZ003": "datetime.datetime.utcnow() used - prefer datetime.datetime.now(tz=datetime.timezone.utc)",
    "DTZ004": "datetime.datetime.utcfromtimestamp() used",
    "DTZ005": "datetime.datetime.now() called without a tz argument",
    "DTZ006": "datetime.datetime.fromtimestamp() called without a tz argument",
    "DTZ007": "datetime.datetime.strptime() without %z must be followed by .replace(tzinfo=)",
    "DTZ011": "datetime.date.today() used - prefer datetime.datetime.now(tz=).date()",
    "DTZ012": "datetime.date.fromtimestamp() used - prefer datetime.datetime.fromtimestamp(tz=).date()",

    # pycodestyle errors
    "E101": "Indentation contains mixed spaces and tabs",
    "E111": "Indentation is not a multiple of four",
    "E112": "Expected an indented block",
    "E113": "Unexpected indentation",
    "E114": "Indentation is not a multiple of four (comment)",
    "E115": "Expected an indented block (comment)",
    "E116": "Unexpected indentation (comment)",
    "E117": "Over-indented",
    "E121": "Continuation line under-indented for hanging indent",
    "E122": "Continuation line missing indentation or outdented",
    "E123": "Closing bracket does not match indentation of opening bracket's line",
    "E124": "Closing bracket does not match visual indentation",
    "E125": "Continuation line with same indent as next logical line",
    "E126": "Continuation line over-indented for hanging indent",
    "E127": "Continuation line over-indented for visual indent",
    "E128": "Continuation line under-indented for visual indent",
    "E129": "Visually indented line with same indent as next logical line",
    "E131": "Continuation line unaligned for hanging indent",
    "E133": "Closing bracket is missing indentation",
    "E201": "Whitespace after '('",
    "E202": "Whitespace before ')'",
    "E203": "Whitespace before ':'",
    "E211": "Whitespace before '('",
    "E221": "Multiple spaces before operator",
    "E222": "Multiple spaces after operator",
    "E223": "Tab before operator",
    "E224": "Tab after operator",
    "E225": "Missing whitespace around operator",
    "E226": "Missing whitespace around arithmetic operator",
    "E227": "Missing whitespace around bitwise or shift operator",
    "E228": "Missing whitespace around modulo operator",
    "E231": "Missing whitespace after ',', ';', or ':'",
    "E241": "Multiple spaces after ','",
    "E242": "Tab after ','",
    "E251": "Unexpected spaces around keyword / parameter equals",
    "E252": "Missing whitespace around parameter equals",
    "E261": "At least two spaces before inline comment",
    "E262": "Inline comment should start with '# '",
    "E265": "Block comment should start with '# '",
    "E266": "Too many leading '#' for block comment",
    "E271": "Multiple spaces after keyword",
    "E272": "Multiple spaces before keyword",
    "E273": "Tab after keyword",
    "E274": "Tab before keyword",
    "E275": "Missing whitespace after keyword",
    "E301": "Expected 1 blank line, found 0",
    "E302": "Expected 2 blank lines, found N",
    "E303": "Too many blank lines",
    "E304": "Blank lines found after function decorator",
    "E305": "Expected 2 blank lines after class or function definition, found N",
    "E306": "Expected 1 blank line before a nested definition, found N",
    "E401": "Multiple imports on one line",
    "E402": "Module level import not at top of file",
    "E501": "Line too long",
    "E502": "The backslash is redundant between brackets",
    "E701": "Multiple statements on one line (colon)",
    "E702": "Multiple statements on one line (semicolon)",
    "E703": "Statement ends with a semicolon",
    "E704": "Multiple statements on one line (def)",
    "E711": "Comparison to None should be 'is None'",
    "E712": "Comparison to True/False should be 'if cond is True:' or 'if cond:'",
    "E713": "Test for membership should be 'not in'",
    "E714": "Test for object identity should be 'is not'",
    "E721": "Do not compare types, for exact checks use `is` or `is not`, for instance checks use `isinstance()`",
    "E722": "Do not use bare 'except'",
    "E731": "Do not assign a lambda expression, use a def",
    "E741": "Ambiguous variable name",
    "E742": "Ambiguous class definition",
    "E743": "Ambiguous function definition",
    "E902": "Tokenization error",
    "E999": "Syntax error in source file",

    # flake8-errmsg
    "EM101": "Exception must not use a string literal, assign to variable first",
    "EM102": "Exception must not use an f-string literal, assign to variable first",
    "EM103": "Exception must not use a .format() string, assign to variable first",

    # eradicate
    "ERA001": "Found commented-out code",

    # flake8-executable
    "EXE001": "Shebang is present but file is not executable",
    "EXE002": "The file is executable but no shebang is present",
    "EXE003": "Shebang should contain python",
    "EXE004": "Shebang should have at most one space before python",
    "EXE005": "Shebang should be at the beginning of the file",

    # Pyflakes
    "F401": "Module imported but unused",
    "F402": "Import shadowed by loop variable",
    "F403": "'from module import *' used; unable to detect undefined names",
    "F404": "Future import(s) name after other statements",
    "F405": "Name may be undefined, or defined from star imports",
    "F406": "'from module import *' only allowed at module level",
    "F407": "Future feature is not defined",
    "F501": "%-format string has invalid format string",
    "F502": "%-format string expected mapping but got sequence",
    "F503": "%-format string expected sequence but got mapping",
    "F504": "%-format string unused named arguments",
    "F505": "%-format string missing required named arguments",
    "F506": "%-format string mixed positional and named arguments",
    "F507": "%-format string mismatch of placeholder and argument count",
    "F508": "%-format string * specifier requires a sequence",
    "F509": "%-format string unsupported format character",
    "F521": ".format(...) has invalid format string",
    "F522": ".format(...) unused named arguments",
    "F523": ".format(...) unused positional arguments",
    "F524": ".format(...) missing argument(s) for placeholder(s)",
    "F525": ".format(...) mixes automatic and manual numbering",
    "F601": "Dictionary key literal repeated",
    "F602": "Dictionary key variable repeated",
    "F621": "Too many expressions in an assignment with star-unpacking",
    "F622": "Two or more starred expressions in an assignment",
    "F631": "Assert test is a non-empty tuple, which is always True",
    "F632": "Use == to compare str, bytes, and int literals",
    "F633": "Use of >> is invalid with print function",
    "F634": "If test is a tuple, which is always True",
    "F701": "A break statement not in a for or while loop",
    "F702": "A continue statement not in a for or while loop",
    "F704": "Yield statement outside function",
    "F706": "Return statement outside function",
    "F707": "An except: block as not the last exception handler",
    "F811": "Redefinition of unused name from line N",
    "F821": "Undefined name",
    "F822": "Undefined name in __all__",
    "F823": "Local variable referenced before assignment",
    "F841": "Local variable is assigned to but never used",
    "F842": "Local variable is annotated but never used",
    "F901": "Raise NotImplemented should be raise NotImplementedError",

    # FastAPI
    "FAST001": "FastAPI route with redundant response_model argument",
    "FAST002": "FastAPI dependency without Annotated",
    "FAST003": "FastAPI route with untyped path parameter",

    # flake8-boolean-trap
    "FBT001": "Boolean-typed positional argument in function definition",
    "FBT002": "Boolean default positional argument in function definition",
    "FBT003": "Boolean positional value in function call",

    # flake8-fixme
    "FIX001": "Line contains FIXME",
    "FIX002": "Line contains TODO",
    "FIX003": "Line contains XXX",
    "FIX004": "Line contains HACK",

    # flynt
    "FLY002": "Consider using f-string instead of string concatenation",

    # flake8-logging-format
    "G001": "Logging statement uses str.format",
    "G002": "Logging statement uses % formatting",
    "G003": "Logging statement uses + concatenation",
    "G004": "Logging statement uses f-string",
    "G010": "Logging statement uses warn instead of warning",
    "G101": "Logging statement uses an extra field that conflicts with a LogRecord field",
    "G201": "Logging .exception(...) should be used inside an exception handler",
    "G202": "Logging statement has redundant exc_info",

    # isort
    "I001": "Import block is un-sorted or un-formatted",
    "I002": "Missing required import",

    # flake8-import-conventions
    "ICN001": "Unconventional import alias",
    "ICN002": "Banned import alias",
    "ICN003": "Banned import from module",

    # flake8-no-pep420
    "INP001": "File is part of an implicit namespace package - add __init__.py",

    # flake8-gettext
    "INT001": "f-string is resolved before function call; consider _(\"string %s\") % arg",
    "INT002": ".format() is resolved before function call; consider _(\"string %s\") % arg",
    "INT003": "printf-style format is resolved before function call",

    # flake8-implicit-str-concat
    "ISC001": "Implicitly concatenated string literals on one line",
    "ISC002": "Implicitly concatenated string literals over continuation line",
    "ISC003": "Explicitly concatenated string should be implicitly concatenated",

    # flake8-logging
    "LOG001": "Use logging.getLogger() to instantiate loggers",
    "LOG002": "Use __name__ with getLogger()",
    "LOG007": "Use exc_info=True within logging.exception()",
    "LOG009": "Use logging.warning instead of deprecated logging.warn",

    # pep8-naming
    "N801": "Class name should use CapWords convention",
    "N802": "Function name should be lowercase",
    "N803": "Argument name should be lowercase",
    "N804": "First argument of a classmethod should be named cls",
    "N805": "First argument of a method should be named self",
    "N806": "Variable in function should be lowercase",
    "N807": "Function name should not start and end with __",
    "N811": "Constant imported as non-constant",
    "N812": "Lowercase imported as non-lowercase",
    "N813": "Camelcase imported as lowercase",
    "N814": "Camelcase imported as constant",
    "N815": "mixedCase variable in class scope",
    "N816": "mixedCase variable in global scope",
    "N817": "CamelCase imported as acronym",
    "N818": "Exception name should be named with an Error suffix",
    "N999": "Invalid module name",

    # NumPy-specific
    "NPY001": "Type alias np.bool is deprecated, use bool instead",
    "NPY002": "Replace legacy np.random.x call with np.random.Generator",
    "NPY003": "Deprecated NumPy type alias used",
    "NPY201": "NumPy 2.0 deprecation check",

    # pandas-vet
    "PD002": "inplace=True should be avoided; it has inconsistent behavior",
    "PD003": ".isna is preferred to .isnull; functionality is equivalent",
    "PD004": ".notna is preferred to .notnull; functionality is equivalent",
    "PD007": ".iloc is preferred to .ix; .ix is deprecated",
    "PD008": ".loc is preferred to .at; use .loc for label-based access",
    "PD009": ".iloc is preferred to .iat; use .iloc for positional access",
    "PD010": ".pivot_table is preferred to .pivot or .unstack",
    "PD011": "Use .to_numpy() instead of .values",
    "PD012": ".read_csv(..., squeeze=True) is deprecated",
    "PD013": ".melt is preferred to .stack; provides same functionality",
    "PD015": "Use .merge method instead of pd.merge function",
    "PD101": "Use .nunique for counting unique values",
    "PD901": "Avoid using 'df' as a variable name; be more descriptive",

    # Perflint
    "PERF101": "Do not cast an iterable to list before iterating over it",
    "PERF102": "When using only the keys/values of a dict use keys()/values()",
    "PERF203": "try-except within a loop incurs performance overhead",
    "PERF401": "Use a list comprehension to create a transformed list",
    "PERF402": "Use list or list.copy to create a copy of a list",
    "PERF403": "Use a dictionary comprehension instead of a for-loop",

    # flake8-pie
    "PIE790": "No unnecessary pass statements",
    "PIE794": "Class field defined multiple times",
    "PIE796": "Non-unique enums",
    "PIE800": "No unnecessary dict kwargs",
    "PIE804": "No unnecessary dict kwargs",
    "PIE807": "Prefer list over useless lambda",
    "PIE808": "Prefer simple range without redundant start argument",
    "PIE810": "Call startswith or endswith once with a tuple",

    # Pylint Convention
    "PLC0105": "TypeVar (or ParamSpec/TypeVarTuple) name does not reflect its covariance",
    "PLC0131": "TypeVar bound or constraint is not a type",
    "PLC0132": "TypeVar (or ParamSpec/TypeVarTuple) name does not match assigned name",
    "PLC0205": "Class __slots__ should be a non-string iterable",
    "PLC0208": "Use a sequence type when iterating over values",
    "PLC0414": "Import alias does not rename original package",
    "PLC0415": "Import outside top-level of file",
    "PLC1901": "Empty string is falsey; use 'not' for empty check",
    "PLC2401": "Variable name contains non-ASCII unicode character",
    "PLC2403": "Module import not at top of file",
    "PLC2701": "Private name imported as public name",
    "PLC2801": "Unnecessary dunder call",
    "PLC3002": "Lambda expression called directly; use inline expression",

    # Pylint Error
    "PLE0100": "__init__ method is a generator",
    "PLE0101": "Explicit return in __init__",
    "PLE0116": "continue not properly in loop",
    "PLE0117": "nonlocal name found without binding",
    "PLE0118": "Name is used prior to global declaration",
    "PLE0241": "Duplicate base class",
    "PLE0302": "The special method expects a specific number of parameters",
    "PLE0303": "__len__ does not return non-negative integer",
    "PLE0304": "__bool__ does not return bool",
    "PLE0305": "__index__ does not return int",
    "PLE0307": "__str__ does not return str",
    "PLE0308": "__bytes__ does not return bytes",
    "PLE0309": "__hash__ does not return int",
    "PLE0604": "Invalid object in __all__, must contain only strings",
    "PLE0605": "Invalid format for __all__, must be list or tuple",
    "PLE0643": "Potential __index__ method returning non-integer",
    "PLE0704": "Bare raise not inside exception handler",
    "PLE1132": "Repeated keyword argument in function call",
    "PLE1141": "Unpacking a dictionary in iteration without calling .items()",
    "PLE1142": "await inside comprehension",
    "PLE1205": "Too many arguments for logging format string",
    "PLE1206": "Not enough arguments for logging format string",
    "PLE1300": "Unsupported format character in format string",
    "PLE1307": "Format type not matching argument type",
    "PLE1310": "Bad str/bytes/bytearray strip argument",
    "PLE1507": "Invalid envvar value",
    "PLE1519": "Singular import from __future__",
    "PLE1700": "yield from statement in async function",
    "PLE2502": "Contains a non-UTF-8 character",
    "PLE2510": "Invalid unescaped character backslash",
    "PLE2512": "Invalid unescaped character quote",
    "PLE2513": "Invalid unescaped character SUB",
    "PLE2514": "Invalid unescaped character ESC",
    "PLE2515": "Invalid unescaped character NUL",
    "PLE4703": "Modified iterator in for loop body",

    # Pylint Refactor
    "PLR0124": "Name compared with itself",
    "PLR0133": "Two constants compared in a comparison",
    "PLR0202": "Method could be a classmethod",
    "PLR0203": "Method could be a staticmethod",
    "PLR0206": "Cannot have defined parameters for property getter",
    "PLR0402": "Use from X import Y instead of import alias",
    "PLR0904": "Too many public methods",
    "PLR0911": "Too many return statements",
    "PLR0912": "Too many branches",
    "PLR0913": "Too many arguments in function definition",
    "PLR0914": "Too many local variables",
    "PLR0915": "Too many statements",
    "PLR0916": "Too many Boolean expressions",
    "PLR0917": "Too many positional arguments",
    "PLR1701": "Merge isinstance calls",
    "PLR1702": "Too many nested blocks",
    "PLR1704": "Redefining argument with the local name",
    "PLR1706": "Use ternary instead of or expression",
    "PLR1711": "Useless return at end of function",
    "PLR1714": "Consider merging multiple comparisons",
    "PLR1722": "Use sys.exit() instead of exit() or quit()",
    "PLR1730": "Replace if statement with min/max call",
    "PLR1733": "Unnecessary dictionary index lookup, use .items()",
    "PLR1736": "Unnecessary list index lookup, use enumerate()",
    "PLR2004": "Magic value used in comparison",
    "PLR2044": "Line with empty comment",
    "PLR5501": "Use else if instead of elif after if block with return",
    "PLR6104": "Non-augmented assignment",
    "PLR6201": "Use a set literal when testing for membership",
    "PLR6301": "Method could be a function or static method",

    # Pylint Warning
    "PLW0108": "Lambda may be unnecessary",
    "PLW0120": "Loop else clause is only executed if loop completes successfully",
    "PLW0127": "Assigning same name to a variable",
    "PLW0128": "Redeclared variable in assignment",
    "PLW0129": "Asserting on an empty sequence/mapping will never raise",
    "PLW0131": "Named expression used without context",
    "PLW0133": "Exception value not used",
    "PLW0177": "Comparison with callable, did you forget parentheses?",
    "PLW0211": "Bad staticmethod argument",
    "PLW0245": "super() call without calling parent's method",
    "PLW0406": "Module import itself",
    "PLW0602": "Using global for a variable not in module scope",
    "PLW0603": "Using the global statement to update a variable",
    "PLW0604": "Using the global statement at module level",
    "PLW0642": "Reassigned self in instance method",
    "PLW0711": "Binary op exception handling",
    "PLW1501": "open() should be called with keyword argument mode",
    "PLW1508": "Invalid type for environment variable default",
    "PLW1509": "Preexec_fn is not safe with threads",
    "PLW1510": "subprocess.run without capture_output or check",
    "PLW1514": "open in text mode without explicit encoding",
    "PLW1641": "__eq__ without __hash__",
    "PLW2101": "Global variable not assigned",
    "PLW2901": "Outer loop variable overwritten by inner assignment target",
    "PLW3201": "Bad dunder method name",
    "PLW3301": "Nested min/max call",

    # flake8-pytest-style
    "PT001": "Use @pytest.fixture() over @pytest.fixture",
    "PT002": "Configuration for fixture should be in pytest.fixture()",
    "PT003": "scope='function' is implied in @pytest.fixture()",
    "PT004": "Fixture does not return anything, add leading underscore",
    "PT005": "Fixture returns a value, remove leading underscore",
    "PT006": "Wrong name(s) type in @pytest.mark.parametrize",
    "PT007": "Wrong values type in @pytest.mark.parametrize",
    "PT008": "Use return_value= instead of lambda for patching",
    "PT009": "Use a regular assert instead of unittest-style assertEqual",
    "PT010": "Set the expected exception in pytest.raises()",
    "PT011": "pytest.raises(...) is too broad, set match parameter",
    "PT012": "pytest.raises() block should contain a single simple statement",
    "PT013": "Found incorrect import of pytest, use simple import pytest",
    "PT014": "Duplicate test ids in @pytest.mark.parametrize",
    "PT015": "Assertion always fails, replace with pytest.fail()",
    "PT016": "No message passed to pytest.fail()",
    "PT017": "Found assertion on exception instead of asserting type",
    "PT018": "Assertion should be broken down into multiple parts",
    "PT019": "Fixture without value is injected as parameter",
    "PT020": "@pytest.yield_fixture is deprecated, use @pytest.fixture",
    "PT021": "Use yield instead of request.addfinalizer",
    "PT022": "No teardown in fixture, use return instead of yield",
    "PT023": "Use @pytest.mark.xxx() over @pytest.mark.xxx",
    "PT024": "pytest.mark.asyncio is unnecessary for fixtures",
    "PT025": "pytest.mark.usefixtures has no effect on fixtures",
    "PT026": "Useless pytest.mark.usefixtures without parameters",
    "PT027": "Use pytest.raises() instead of unittest-style method",

    # flake8-use-pathlib
    "PTH100": "os.path.abspath() should be replaced by Path.resolve()",
    "PTH101": "os.chmod() should be replaced by Path.chmod()",
    "PTH102": "os.mkdir() should be replaced by Path.mkdir()",
    "PTH103": "os.makedirs() should be replaced by Path.mkdir(parents=True)",
    "PTH104": "os.rename() should be replaced by Path.rename()",
    "PTH105": "os.replace() should be replaced by Path.replace()",
    "PTH106": "os.rmdir() should be replaced by Path.rmdir()",
    "PTH107": "os.remove() should be replaced by Path.unlink()",
    "PTH108": "os.unlink() should be replaced by Path.unlink()",
    "PTH109": "os.getcwd() should be replaced by Path.cwd()",
    "PTH110": "os.path.exists() should be replaced by Path.exists()",
    "PTH111": "os.path.expanduser() should be replaced by Path.expanduser()",
    "PTH112": "os.path.isdir() should be replaced by Path.is_dir()",
    "PTH113": "os.path.isfile() should be replaced by Path.is_file()",
    "PTH114": "os.path.islink() should be replaced by Path.is_symlink()",
    "PTH115": "os.readlink() should be replaced by Path.readlink()",
    "PTH116": "os.stat() should be replaced by Path.stat()",
    "PTH117": "os.path.isabs() should be replaced by Path.is_absolute()",
    "PTH118": "os.path.join() should be replaced by Path / operator",
    "PTH119": "os.path.basename() should be replaced by Path.name",
    "PTH120": "os.path.dirname() should be replaced by Path.parent",
    "PTH121": "os.path.samefile() should be replaced by Path.samefile()",
    "PTH122": "os.path.splitext() should be replaced by Path.suffix",
    "PTH123": "open() should be replaced by Path.open()",
    "PTH124": "py.path.local is deprecated; use pathlib.Path",
    "PTH201": "pathlib.Path().resolve() should be avoided in favor of cwd() or resolve()",
    "PTH202": "os.path.getsize() should be replaced by Path.stat().st_size",
    "PTH203": "os.path.getatime() should be replaced by Path.stat().st_atime",
    "PTH204": "os.path.getmtime() should be replaced by Path.stat().st_mtime",
    "PTH205": "os.path.getctime() should be replaced by Path.stat().st_ctime",
    "PTH206": "os.path.sep should be replaced by Path's functionality",
    "PTH207": "glob should be replaced by Path.glob() or Path.rglob()",

    # flake8-pyi
    "PYI001": "Name of private TypeVar should start with _",
    "PYI002": "Complex if statement in stub, use @overload",
    "PYI003": "Unrecognized sys.version_info check",
    "PYI004": "Version outdated, remove if block",
    "PYI005": "Version check invalid, remove if block",
    "PYI006": "Use only < and >= for version checks",
    "PYI007": "Unrecognized sys.platform check",
    "PYI008": "Unrecognized platform; expected linux, darwin, or win32",
    "PYI009": "Use ... instead of pass in stub",
    "PYI010": "Use ... instead of pass in function body stub",
    "PYI011": "Typed argument should not have default",
    "PYI012": "Class body should not contain pass",
    "PYI013": "Non-empty class body should not contain ...",
    "PYI014": "Use ... in default values for stubs",
    "PYI015": "Attribute must not have a default value",
    "PYI016": "Duplicate union member",
    "PYI017": "Complex if/else in stub, use @overload",
    "PYI018": "Private TypeVar is never used",
    "PYI019": "Methods should use Self instead of custom TypeVar",
    "PYI020": "Quoted annotations should not be included in stubs",
    "PYI021": "Docstrings should not be included in stubs",
    "PYI024": "Use typing.NamedTuple instead of collections.namedtuple",
    "PYI025": "Use from collections.abc import Set as AbstractSet",
    "PYI026": "Use typing_extensions.TypeAlias for type aliases",
    "PYI029": "Defining __str__ or __repr__ in a stub is almost always useless",
    "PYI030": "Multiple literal members in a union; use Literal[...]",
    "PYI032": "Use object instead of Any for __eq__/__ne__ second parameter",
    "PYI033": "Do not use type comments in stubs",
    "PYI034": "__new__ should return Self",
    "PYI035": "__init__ should return None",
    "PYI036": "__exit__ first argument should be type[BaseException] | None",
    "PYI041": "Use float instead of int | float in type unions",
    "PYI042": "Type alias names should use CamelCase",
    "PYI043": "Private type alias should not be suffixed with T",
    "PYI044": "from __future__ import annotations has no effect in stubs",
    "PYI045": "__iter__ methods should return an Iterator, not Iterable",
    "PYI046": "Private protocol never used",
    "PYI047": "Private TypeAlias never used",
    "PYI048": "Stub function body with docstring should contain ...",
    "PYI049": "Private TypedDict never used",
    "PYI050": "Never annotate return type as NoReturn, use None",
    "PYI051": "Literal[\"string\"] should be preferred over typing aliases",
    "PYI052": "Need default value for non-abstract method",
    "PYI053": "String and bytes literals longer than 50 characters not allowed",
    "PYI054": "Numeric literals with more than 10 digits not allowed",
    "PYI055": "Redundant union of None and type[None]",
    "PYI056": "Redundant literal in union, use plain type",
    "PYI057": "Use ByteString as a type is deprecated",
    "PYI058": "Return type should be Generator, not Iterator",
    "PYI059": "Use Generic instead of __class_getitem__",
    "PYI062": "Duplicate literal member",
    "PYI063": "Use PEP 570 syntax for positional-only arguments",
    "PYI064": "Redundant numeric literal in union",
    "PYI066": "Use int instead of str in __version__",

    # flake8-quotes
    "Q000": "Single quotes found but double quotes preferred",
    "Q001": "Single quote multiline found but double quotes preferred",
    "Q002": "Single quote docstring found but double quotes preferred",
    "Q003": "Change outer quotes to avoid escaping inner quotes",
    "Q004": "Unnecessary escape on inner quote character",

    # flake8-return
    "RET501": "Do not explicitly return None in function if it is the only possible value",
    "RET502": "Do not implicitly return None in function able to return non-None value",
    "RET503": "Missing explicit return at the end of function able to return non-None value",
    "RET504": "Unnecessary assignment before return statement",
    "RET505": "Unnecessary else after return statement",
    "RET506": "Unnecessary else after raise statement",
    "RET507": "Unnecessary else after continue statement",
    "RET508": "Unnecessary else after break statement",

    # flake8-raise
    "RSE102": "Unnecessary parentheses on raised exception",

    # Ruff-specific
    "RUF001": "String contains ambiguous unicode character",
    "RUF002": "Docstring contains ambiguous unicode character",
    "RUF003": "Comment contains ambiguous unicode character",
    "RUF005": "Consider iterable unpacking instead of concatenation",
    "RUF006": "Store a reference to the return value of asyncio.create_task",
    "RUF007": "Prefer itertools.pairwise() over zip() when iterating over successive pairs",
    "RUF008": "Do not use mutable default values for dataclass attributes",
    "RUF009": "Do not perform function call in dataclass defaults",
    "RUF010": "Use explicit conversion flag",
    "RUF011": "Dictionary comprehension with static key",
    "RUF012": "Mutable class attributes should be annotated with typing.ClassVar",
    "RUF013": "PEP 484 prohibits implicit Optional",
    "RUF015": "Prefer next(...) over single element slice",
    "RUF016": "Slice in indexed access should not be type-confused",
    "RUF017": "Avoid quadratic list summation",
    "RUF018": "Avoid assignment expressions in assert statements",
    "RUF019": "Unnecessary key check before dictionary access",
    "RUF020": "typing.Never | T is equivalent to T",
    "RUF021": "Parenthesize a and b when chaining",
    "RUF022": "__all__ is not sorted",
    "RUF023": "__slots__ is not sorted",
    "RUF024": "Do not pass mutable objects as values in dict.fromkeys",
    "RUF025": "Unnecessary dict comprehension, use dict.fromkeys",
    "RUF026": "default_factory is a positional-only argument",
    "RUF027": "Possible f-string without an f prefix",
    "RUF028": "Invalid formatter suppression comment",
    "RUF029": "Function is declared async but has no await",
    "RUF030": "assert x, False instead of assert x, 'message'",
    "RUF031": "Incorrectly parenthesized tuple in subscript",
    "RUF032": "Decimal() called with a float literal",
    "RUF033": "__post_init__ should not have any parameters",
    "RUF034": "Useless if-else condition",
    "RUF100": "Unused noqa directive",
    "RUF101": "Redirected noqa directive",
    "RUF200": "Failed to parse pyproject.toml",

    # flake8-bandit
    "S101": "Use of assert detected",
    "S102": "Use of exec detected",
    "S103": "Chmod setting a permissive mask on file or directory",
    "S104": "Possible binding to all interfaces",
    "S105": "Possible hardcoded password",
    "S106": "Possible hardcoded password in function argument",
    "S107": "Possible hardcoded password in function default",
    "S108": "Probable insecure usage of temp file/directory",
    "S110": "try-except-pass detected, consider logging the exception",
    "S112": "try-except-continue detected, consider logging the exception",
    "S113": "Probable use of requests call without timeout",
    "S201": "Flask app with debug=True is insecure",
    "S202": "Uses of tarfile.extractall()",
    "S301": "Pickle and modules that wrap it can be unsafe",
    "S302": "Deserialization with marshal module is possibly dangerous",
    "S303": "Use of insecure MD2, MD4, MD5, or SHA1 hash function",
    "S304": "Use of insecure cipher or cipher mode",
    "S305": "Use of insecure cipher mode",
    "S306": "Use of insecure and deprecated function mktemp()",
    "S307": "Use of possibly insecure function eval()",
    "S308": "Use of mark_safe() may expose cross-site scripting vulnerabilities",
    "S310": "Audit url open for permitted schemes",
    "S311": "Standard pseudo-random generators are not suitable for security",
    "S312": "Telnet-related functions are being called",
    "S313": "Using xml.etree.cElementTree.parse to parse untrusted XML is insecure",
    "S314": "Using xml.etree.ElementTree.parse to parse untrusted XML is insecure",
    "S315": "Using xml.expat.xmlparser to parse untrusted XML is insecure",
    "S316": "Using xml.dom.expatbuilder to parse untrusted XML is insecure",
    "S317": "Using xml.sax to parse untrusted XML is insecure",
    "S318": "Using xml.dom.minidom to parse untrusted XML is insecure",
    "S319": "Using xml.dom.pulldom to parse untrusted XML is insecure",
    "S320": "Using lxml.etree to parse untrusted XML is insecure",
    "S321": "FTP-related functions are being called",
    "S323": "By default, SSL/TLS contexts allow insecure protocols",
    "S324": "Probable use of insecure hash functions in hashlib",
    "S401": "telnetlib module is deprecated and insecure",
    "S402": "ftplib module is deprecated and insecure",
    "S403": "Consider possible security implications of pickle module",
    "S404": "Consider possible security implications of subprocess module",
    "S405": "Using xml.etree is insecure when parsing untrusted XML",
    "S406": "Using xml.sax is insecure when parsing untrusted XML",
    "S407": "Using xml.expat.defusedexpat is insecure when parsing untrusted XML",
    "S408": "Using xml.dom.minidom is insecure when parsing untrusted XML",
    "S409": "Using xml.dom.pulldom is insecure when parsing untrusted XML",
    "S410": "lxml detected, consider using defusedxml",
    "S411": "Using xmlrpc is insecure, use zeep or requests instead",
    "S412": "Using httpoxy-affected code is insecure",
    "S413": "Using pycrypto is deprecated, use pyca/cryptography instead",
    "S415": "pyghmi detected, consider using OpenBMC instead",
    "S501": "Requests library is being called without verify=True",
    "S502": "Requests library is being called with verify=False",
    "S503": "Requests library is being called without timeout",
    "S504": "Requests library is being called with cert not verified",
    "S505": "Use of a weak cryptographic key",
    "S506": "Unsafe yaml loading",
    "S507": "Missing host key verification",
    "S508": "Snmp insecure version",
    "S509": "Possible wildcard injection in os.system/subprocess calls",
    "S601": "Possible shell injection via Paramiko call",
    "S602": "subprocess call with shell=True seems safe, but may be changed in future",
    "S603": "subprocess call - check for execution of untrusted input",
    "S604": "Any function call with shell=True is a potential security risk",
    "S605": "Starting a process with a shell is a security risk",
    "S606": "Starting a process without a shell (safer)",
    "S607": "Starting a process with a partial executable path",
    "S608": "Possible SQL injection vector through string-based query construction",
    "S609": "Possible wildcard injection in call",
    "S610": "Use of Django extra() can result in SQL injection",
    "S611": "Use of Django RawSQL() can result in SQL injection",
    "S612": "Using logging.basicConfig to set up root logger is insecure",
    "S701": "Using jinja2 templates with autoescape=False is dangerous",
    "S702": "Using mako templates is dangerous",
    "S703": "Jinja2 environment without autoescape enabled",
    "S704": "Using Django mark_safe can lead to XSS vulnerabilities",

    # flake8-simplify
    "SIM101": "Multiple isinstance calls for expr, merge into a single call",
    "SIM102": "Use a single if statement instead of nested if statements",
    "SIM103": "Return the condition directly instead of if-else-block",
    "SIM105": "Use contextlib.suppress(...) instead of try-except-pass",
    "SIM107": "Don't use return in try/except and finally",
    "SIM108": "Use ternary operator instead of if-else block",
    "SIM109": "Use tuple instead of multiple or comparisons",
    "SIM110": "Use any(...) instead of for loop",
    "SIM111": "Use all(...) instead of for loop",
    "SIM112": "Use capitalized environment variable",
    "SIM113": "Use enumerate for index variable in for loop",
    "SIM114": "Combine if branches using logical or operator",
    "SIM115": "Use context manager for opening files",
    "SIM116": "Use a dictionary instead of consecutive if statements",
    "SIM117": "Use a single with statement with multiple contexts",
    "SIM118": "Use key in dict instead of key in dict.keys()",
    "SIM201": "Use not ... instead of not (... == ...)",
    "SIM202": "Use not ... instead of not (... != ...)",
    "SIM208": "Use not ... instead of not (not ...)",
    "SIM210": "Use bool(...) instead of True if ... else False",
    "SIM211": "Use not ... instead of False if ... else True",
    "SIM212": "Use ... if ... else ... instead of ... if not ... else ...",
    "SIM220": "Use False instead of ... and not ...",
    "SIM221": "Use True instead of ... or not ...",
    "SIM222": "Use True instead of ... or True",
    "SIM223": "Use False instead of ... and False",
    "SIM300": "Yoda conditions are discouraged, use ... instead",
    "SIM401": "Use dict.get(...) instead of if-else block",
    "SIM904": "Initialize dictionary directly instead of using .update()",
    "SIM905": "Split string using .split() instead of slicing",
    "SIM907": "Use Optional[...] instead of Union[..., None]",
    "SIM908": "Use dict.get() instead of if-else block for default",
    "SIM910": "Use dict.get(key) instead of dict.get(key, None)",
    "SIM911": "Use zip(...) instead of dict(zip(...))",

    # flake8-self
    "SLF001": "Private member accessed outside of the owning class",

    # flake8-slots
    "SLOT000": "Subclass of str should define __slots__",
    "SLOT001": "Subclass of tuple should define __slots__",
    "SLOT002": "Subclass of namedtuple should define __slots__",

    # flake8-debugger
    "T100": "Trace or debug statement found",

    # flake8-print
    "T201": "print found",
    "T203": "pprint found",

    # flake8-type-checking
    "TC001": "Move application import into a type-checking block",
    "TC002": "Move third-party import into a type-checking block",
    "TC003": "Move standard library import into a type-checking block",
    "TC004": "Move import out of type-checking block, used at runtime",
    "TC005": "Found empty type-checking block",
    "TC006": "Add quotes to type expression in typing.cast()",
    "TC007": "Use quotes for type expression in type alias",
    "TC008": "Use quotes for type expression in type alias with TypeAlias",
    "TC010": "Use | None instead of Optional",

    # flake8-todos
    "TD001": "Invalid TODO tag; use TODO",
    "TD002": "Missing author in TODO; try: # TODO(author): ...",
    "TD003": "Missing issue link on the line following this TODO",
    "TD004": "Missing colon in TODO",
    "TD005": "Missing issue description after TODO",
    "TD006": "Invalid TODO capitalization",
    "TD007": "Missing space after colon in TODO",

    # flake8-tidy-imports
    "TID251": "Banned import",
    "TID252": "Prefer absolute imports over relative imports from parent modules",
    "TID253": "Banned module level import",

    # tryceratops
    "TRY002": "Create your own exception instead of raising ValueError/TypeError",
    "TRY003": "Avoid specifying long messages outside the exception class",
    "TRY004": "Prefer TypeError exception for invalid type",
    "TRY200": "Use raise from to specify exception cause",
    "TRY201": "Simply use raise without specifying exception name",
    "TRY203": "Except handler should use raise ... from exc",
    "TRY300": "Consider moving this statement to an else block",
    "TRY301": "Abstract raise to an inner function",
    "TRY302": "Remove exception handler; error is immediately re-raised",
    "TRY400": "Use logging.exception instead of logging.error",
    "TRY401": "Redundant exception object included in logging.exception call",

    # pyupgrade
    "UP001": "Unnecessary (double) parentheses",
    "UP003": "Use {} instead of type([])",
    "UP004": "Class inherits from object",
    "UP005": "Deprecated unittest alias",
    "UP006": "Use list instead of typing.List in type annotations",
    "UP007": "Use X | Y for type union",
    "UP008": "Use super() instead of super(__class__, self)",
    "UP009": "UTF-8 encoding declaration is unnecessary",
    "UP010": "Unnecessary __future__ import for target Python version",
    "UP011": "Unnecessary parentheses to functools.lru_cache",
    "UP012": "Unnecessary call to encode as UTF-8",
    "UP013": "Convert class to use __class_getitem__",
    "UP014": "Convert from TypedDict functional syntax to class syntax",
    "UP015": "Unnecessary mode argument to open for reading text file",
    "UP017": "Use datetime.UTC alias",
    "UP018": "Unnecessary call to primitive type constructor",
    "UP019": "Deprecated typing.Text alias, use str",
    "UP020": "Use builtin open",
    "UP021": "Replace universal newlines with text",
    "UP022": "Sending stdout/stderr to os.devnull is unnecessary",
    "UP023": "rewrite cElementTree alias to ElementTree",
    "UP024": "Replace aliased errors with OSError",
    "UP025": "Remove unicode literals from strings",
    "UP026": "Replace mock module with unittest.mock",
    "UP027": "Replace unpacked list comprehension with a generator expression",
    "UP028": "Replace yield in for loop with yield from",
    "UP029": "Unnecessary builtin import",
    "UP030": "Use implicit references for positional format fields",
    "UP031": "Use format specifiers instead of percent format",
    "UP032": "Use f-string instead of format call",
    "UP033": "Use @functools.cache instead of @functools.lru_cache(maxsize=None)",
    "UP034": "Avoid extraneous parentheses",
    "UP035": "Import from collections.abc, not collections",
    "UP036": "Version block is outdated for minimum Python version",
    "UP037": "Remove quotes from type annotation",
    "UP038": "Use X | Y in isinstance call instead of (X, Y)",
    "UP039": "Unnecessary parentheses after class definition",
    "UP040": "Type alias uses TypeAlias annotation instead of type statement",
    "UP041": "Replace aliased errors with TimeoutError",
    "UP042": "Replace PEP 604 union with TypeAlias annotation",
    "UP043": "Unnecessary default type argument",
    "UP044": "Replace non-PEP-585 types with generics",
    "UP045": "Replace Optional[X] with X | None",
    "UP046": "Use TypeAliasType instead of TypeAlias annotation",

    # pycodestyle warnings
    "W191": "Indentation contains tabs",
    "W291": "Trailing whitespace",
    "W292": "No newline at end of file",
    "W293": "Blank line contains whitespace",
    "W391": "Blank line at end of file",
    "W503": "Line break before binary operator",
    "W504": "Line break after binary operator",
    "W505": "Doc line too long",
    "W605": "Invalid escape sequence",

    # flake8-2020
    "YTT101": "sys.version[:3] referenced (python3.10)",
    "YTT102": "sys.version[2] referenced (python3.10)",
    "YTT103": "sys.version compared to string (python3.10)",
    "YTT201": "sys.version_info[0] == 3 referenced (python4)",
    "YTT202": "six.PY3 referenced (python4)",
    "YTT203": "sys.version_info[1] compared to integer (python4)",
    "YTT204": "sys.version_info.minor compared to integer (python4)",
    "YTT301": "sys.version[0] referenced (python10)",
    "YTT302": "sys.version compared to string (python10)",
    "YTT303": "sys.version[:1] referenced (python10)",

    # ==========================================================================
    # Biome Rules
    # ==========================================================================
    "noAccumulatingSpread": "Prevent inefficient spread operations",
    "noConstEnum": "Disallow const enum declarations",
    "noDelete": "Prevent use of delete operator",
    "noDuplicateObjectKeys": "Detect duplicate keys in object literals",
    "noDynamicNamespaceImportAccess": "Prevent dynamic namespace import access",
    "noEmptyTypeParameters": "Disallow empty type parameters",
    "noEnum": "Discourage enum usage",
    "noEvolvingTypes": "Flag types that evolve unexpectedly",
    "noExportedImports": "Prevent re-exporting imports",
    "noGlobalIsFinite": "Avoid global isFinite function",
    "noGlobalIsNan": "Avoid global isNaN function",
    "noImplicitAnyLet": "Require explicit types for let declarations",
    "noInvalidNewBuiltin": "Prevent invalid builtin instantiation",
    "noRedundantUseStrict": "Remove redundant use strict directives",
    "noRenderReturnValue": "Disallow using render return values",
    "noShoutyConstants": "Discourage ALL_CAPS variable names",
    "noSuspiciousSemicolonInJsx": "Catch suspicious semicolons in JSX",
    "noSvgWithoutTitle": "Require title elements in SVG",
    "noUnnecessaryContinue": "Remove unnecessary continue statements",
    "noUnusedFunctionParameters": "Flag unused function parameters",
    "noUnusedTemplateLiteral": "Detect unused template literals",
    "noUselessStringRaw": "Remove unnecessary raw string prefix",
    "noValueAtRule": "Disallow @value at-rules in CSS",
    "noVoidTypeReturn": "Prevent void return type misuse",
    "useImportExtensions": "Enforce explicit file extensions in imports",
    "useNodeAssertStrict": "Use Node.js assert.strict API",
    "useShorthandArrayType": "Prefer T[] over Array<T>",
    "useSimpleNumberKeys": "Use simple number keys in objects",
    "useSimplifiedLogicExpression": "Simplify logical expressions",
    "useSingleCaseStatement": "Avoid single-case switch statements",
    "useSortedClasses": "Enforce sorted class order",
    "useStrictMode": "Require strict mode",
    "useTopLevelRegex": "Place regex at module level",
    "noConsole": "Disallow console statements",
    "noDebugger": "Disallow debugger statements",
    "noDoubleEquals": "Require === and !== instead of == and !=",
    "useConst": "Require const for variables never reassigned",
    "noVar": "Require let or const instead of var",
    "noUnusedVariables": "Disallow unused variables",
    "noExplicitAny": "Disallow the any type",
    "useSingleVarDeclarator": "Disallow multiple variable declarations",
    "noNonNullAssertion": "Disallow non-null assertions (!)",
}
# fmt: on


@dataclass
class DiscoveredRule:
    """A rule discovered from a linter config."""

    linter: str
    rule_id: str
    description: str
    severity: str  # "error", "warn", "off"
    category: str  # convention category
    scope: list[str] = field(default_factory=list)
    pattern: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryResult:
    """Result of convention discovery from config files."""

    linter: str
    config_path: str
    rules: list[DiscoveredRule]
    errors: list[str] = field(default_factory=list)


class ConventionDiscovery:
    """Discovers conventions from linter configuration files."""

    def __init__(self, project_root: Path):
        self.root = project_root

    def discover_all(self) -> list[DiscoveryResult]:
        """Discover conventions from all detected linter configs."""
        results: list[DiscoveryResult] = []

        for linter, config_files in LINTER_CONFIG_FILES.items():
            for config_file in config_files:
                config_path = self.root / config_file
                if config_path.exists():
                    try:
                        result = self._parse_config(linter, config_path)
                        if result.rules:
                            results.append(result)
                    except Exception as e:
                        logger.warning(
                            "failed to parse linter config",
                            linter=linter,
                            path=str(config_path),
                            error=str(e),
                        )

        return results

    def _parse_config(self, linter: str, config_path: Path) -> DiscoveryResult:
        """Parse a linter config file and extract rules."""
        content = config_path.read_text()

        parsers = {
            "eslint": self._parse_eslint,
            "biome": self._parse_biome,
            "ruff": self._parse_ruff,
            "prettier": self._parse_prettier,
            "oxlint": self._parse_oxlint,
        }

        parser = parsers.get(linter)
        if parser:
            return parser(config_path, content)

        return DiscoveryResult(
            linter=linter, config_path=str(config_path), rules=[]
        )

    def _parse_eslint(self, config_path: Path, content: str) -> DiscoveryResult:
        """Parse ESLint configuration."""
        rules: list[DiscoveredRule] = []
        errors: list[str] = []

        try:
            # handle different config formats
            if config_path.suffix in (".json", ""):
                # try to strip comments for .eslintrc
                content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)
                content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
                config = json.loads(content)
            elif config_path.suffix in (".yaml", ".yml"):
                import yaml

                config = yaml.safe_load(content)
            else:
                # .js/.cjs/.mjs - can't easily parse, skip
                return DiscoveryResult(
                    linter="eslint",
                    config_path=str(config_path),
                    rules=[],
                    errors=["JavaScript config format not supported"],
                )

            # extract rules from config
            eslint_rules = config.get("rules", {})
            for rule_id, rule_config in eslint_rules.items():
                # normalize rule config (can be string, number, or array)
                if isinstance(rule_config, (str, int)):
                    severity = self._normalize_severity(rule_config)
                    options = {}
                elif isinstance(rule_config, list) and rule_config:
                    severity = self._normalize_severity(rule_config[0])
                    options = rule_config[1] if len(rule_config) > 1 else {}
                else:
                    continue

                if severity == "off":
                    continue

                rules.append(
                    DiscoveredRule(
                        linter="eslint",
                        rule_id=rule_id,
                        description=RULE_DESCRIPTIONS.get(
                            rule_id, f"ESLint rule: {rule_id}"
                        ),
                        severity=severity,
                        category=self._categorize_rule(rule_id),
                        scope=["context:frontend"],
                        config=options if isinstance(options, dict) else {},
                    )
                )

        except Exception as e:
            errors.append(str(e))

        return DiscoveryResult(
            linter="eslint",
            config_path=str(config_path),
            rules=rules,
            errors=errors,
        )

    def _parse_biome(self, config_path: Path, content: str) -> DiscoveryResult:
        """Parse Biome configuration."""
        rules: list[DiscoveredRule] = []
        errors: list[str] = []

        try:
            # strip comments for jsonc
            content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)
            content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
            config = json.loads(content)

            # biome organizes rules by category
            linter_config = config.get("linter", {})
            rules_config = linter_config.get("rules", {})

            for category, category_rules in rules_config.items():
                if not isinstance(category_rules, dict):
                    continue

                for rule_id, rule_config in category_rules.items():
                    if isinstance(rule_config, str):
                        severity = rule_config.lower()
                    elif isinstance(rule_config, dict):
                        severity = rule_config.get("level", "warn").lower()
                    else:
                        continue

                    if severity == "off":
                        continue

                    rules.append(
                        DiscoveredRule(
                            linter="biome",
                            rule_id=rule_id,
                            description=RULE_DESCRIPTIONS.get(
                                rule_id, f"Biome rule: {rule_id}"
                            ),
                            severity="error" if severity == "error" else "warn",
                            category=self._categorize_biome_category(category),
                            scope=["context:frontend"],
                        )
                    )

        except Exception as e:
            errors.append(str(e))

        return DiscoveryResult(
            linter="biome",
            config_path=str(config_path),
            rules=rules,
            errors=errors,
        )

    def _parse_ruff(self, config_path: Path, content: str) -> DiscoveryResult:
        """Parse Ruff configuration."""
        rules: list[DiscoveredRule] = []
        errors: list[str] = []

        try:
            if config_path.name == "pyproject.toml":
                import tomllib

                config = tomllib.loads(content)
                ruff_config = config.get("tool", {}).get("ruff", {})
            else:
                import tomllib

                ruff_config = tomllib.loads(content)

            # get select/ignore rules
            lint_config = ruff_config.get("lint", ruff_config)
            selected = lint_config.get("select", [])
            ignored = set(lint_config.get("ignore", []))

            for rule_id in selected:
                if rule_id in ignored:
                    continue

                # handle rule prefixes (E, F, I, UP, etc.)
                rules.append(
                    DiscoveredRule(
                        linter="ruff",
                        rule_id=rule_id,
                        description=RULE_DESCRIPTIONS.get(
                            rule_id, f"Ruff rule: {rule_id}"
                        ),
                        severity="error",
                        category=self._categorize_ruff_rule(rule_id),
                        scope=["context:backend"],
                    )
                )

        except Exception as e:
            errors.append(str(e))

        return DiscoveryResult(
            linter="ruff",
            config_path=str(config_path),
            rules=rules,
            errors=errors,
        )

    def _parse_prettier(
        self, config_path: Path, content: str
    ) -> DiscoveryResult:
        """Parse Prettier configuration."""
        rules: list[DiscoveredRule] = []
        errors: list[str] = []

        try:
            if config_path.suffix == ".js":
                return DiscoveryResult(
                    linter="prettier",
                    config_path=str(config_path),
                    rules=[],
                    errors=["JavaScript config format not supported"],
                )

            config = json.loads(content)

            # prettier options -> conventions
            option_mappings = {
                "semi": ("convention:style", "Require semicolons"),
                "singleQuote": ("convention:style", "Use single quotes"),
                "tabWidth": ("convention:style", "Indentation width"),
                "useTabs": ("convention:style", "Use tabs for indentation"),
                "trailingComma": ("convention:style", "Trailing comma style"),
                "bracketSpacing": (
                    "convention:style",
                    "Spaces inside object braces",
                ),
                "arrowParens": (
                    "convention:style",
                    "Arrow function parentheses",
                ),
                "printWidth": ("convention:style", "Maximum line width"),
            }

            for option, (category, desc) in option_mappings.items():
                if option in config:
                    value = config[option]
                    rules.append(
                        DiscoveredRule(
                            linter="prettier",
                            rule_id=option,
                            description=f"{desc}: {value}",
                            severity="warn",
                            category=category,
                            scope=["context:frontend"],
                            config={option: value},
                        )
                    )

        except Exception as e:
            errors.append(str(e))

        return DiscoveryResult(
            linter="prettier",
            config_path=str(config_path),
            rules=rules,
            errors=errors,
        )

    def _parse_oxlint(self, config_path: Path, content: str) -> DiscoveryResult:
        """Parse OxLint configuration."""
        rules: list[DiscoveredRule] = []
        errors: list[str] = []

        try:
            config = json.loads(content)

            # oxlint uses similar format to eslint
            oxlint_rules = config.get("rules", {})
            for rule_id, rule_config in oxlint_rules.items():
                if isinstance(rule_config, str):
                    severity = rule_config.lower()
                else:
                    continue

                if severity == "off":
                    continue

                rules.append(
                    DiscoveredRule(
                        linter="oxlint",
                        rule_id=rule_id,
                        description=RULE_DESCRIPTIONS.get(
                            rule_id, f"OxLint rule: {rule_id}"
                        ),
                        severity="error" if severity == "error" else "warn",
                        category=self._categorize_rule(rule_id),
                        scope=["context:frontend"],
                    )
                )

        except Exception as e:
            errors.append(str(e))

        return DiscoveryResult(
            linter="oxlint",
            config_path=str(config_path),
            rules=rules,
            errors=errors,
        )

    def _normalize_severity(self, value: str | int) -> str:
        """Normalize severity to error/warn/off."""
        if isinstance(value, int):
            return {0: "off", 1: "warn", 2: "error"}.get(value, "off")
        value = str(value).lower()
        if value in ("error", "2"):
            return "error"
        if value in ("warn", "warning", "1"):
            return "warn"
        return "off"

    def _categorize_rule(self, rule_id: str) -> str:
        """Map ESLint-style rule to convention category."""
        rule_lower = rule_id.lower()

        if any(
            x in rule_lower
            for x in ["security", "xss", "injection", "eval", "exec"]
        ):
            return "convention:security"
        if any(x in rule_lower for x in ["perf", "performance", "optimize"]):
            return "convention:performance"
        if any(x in rule_lower for x in ["test", "spec", "jest", "mocha"]):
            return "convention:testing"
        if any(x in rule_lower for x in ["type", "typescript", "ts/"]):
            return "convention:pattern"
        if any(
            x in rule_lower
            for x in ["indent", "quote", "semi", "comma", "space", "style"]
        ):
            return "convention:style"
        if any(
            x in rule_lower
            for x in ["naming", "camel", "pascal", "snake", "case"]
        ):
            return "convention:naming"

        return "convention:style"

    def _categorize_biome_category(self, category: str) -> str:
        """Map Biome category to convention category."""
        mappings = {
            "correctness": "convention:pattern",
            "suspicious": "convention:security",
            "style": "convention:style",
            "complexity": "convention:pattern",
            "performance": "convention:performance",
            "security": "convention:security",
            "a11y": "convention:accessibility",
            "nursery": "convention:style",
        }
        return mappings.get(category.lower(), "convention:style")

    def _categorize_ruff_rule(self, rule_id: str) -> str:
        """Map Ruff rule prefix to convention category."""
        prefix = rule_id.rstrip("0123456789")

        mappings = {
            "E": "convention:style",  # pycodestyle errors
            "W": "convention:style",  # pycodestyle warnings
            "F": "convention:pattern",  # Pyflakes
            "I": "convention:style",  # isort
            "UP": "convention:pattern",  # pyupgrade
            "S": "convention:security",  # flake8-bandit
            "B": "convention:pattern",  # flake8-bugbear
            "C": "convention:pattern",  # flake8-comprehensions
            "SIM": "convention:pattern",  # flake8-simplify
            "ARG": "convention:pattern",  # flake8-unused-arguments
            "PTH": "convention:pattern",  # flake8-use-pathlib
            "ERA": "convention:style",  # flake8-eradicate
            "PL": "convention:pattern",  # Pylint
            "RUF": "convention:pattern",  # Ruff-specific
            "N": "convention:naming",  # pep8-naming
            "D": "convention:documentation",  # pydocstyle
            "ANN": "convention:pattern",  # flake8-annotations
            "T": "convention:testing",  # flake8-pytest-style
        }

        return mappings.get(prefix, "convention:style")


def discover_and_import(
    project_root: Path,
    convention_manager: Any,
    org_id: str | None = None,
) -> dict[str, int]:
    """Discover conventions from linter configs and import them.

    Args:
        project_root: Project root directory
        convention_manager: ConventionManager instance
        org_id: Optional org ID for imported conventions

    Returns:
        Stats with counts per linter
    """
    discovery = ConventionDiscovery(project_root)
    results = discovery.discover_all()

    # get existing convention names to avoid duplicates
    existing = {c.name for c in convention_manager.list(limit=10000)}

    stats: dict[str, int] = {}

    for result in results:
        added = 0
        for rule in result.rules:
            name = f"{result.linter}/{rule.rule_id}"
            if name in existing:
                continue  # skip duplicates

            try:
                priority = (
                    "required" if rule.severity == "error" else "recommended"
                )
                convention_manager.add(
                    name=name,
                    description=rule.description,
                    category=rule.category,
                    scope=rule.scope,
                    priority=priority,
                    pattern=rule.pattern,
                    tags=[result.linter, rule.severity],
                    org_id=org_id,
                )
                added += 1
                existing.add(name)  # track newly added
            except Exception as e:
                logger.debug(
                    "failed to add discovered convention",
                    rule=rule.rule_id,
                    error=str(e),
                )

        stats[result.linter] = added
        logger.info(
            "imported conventions from linter",
            linter=result.linter,
            config=result.config_path,
            rules_imported=added,
        )

    return stats
