const js = require('@eslint/js');
const globals = require('globals');
const prettier = require('eslint-plugin-prettier');
const configPrettier = require('eslint-config-prettier');

module.exports = [
    js.configs.recommended,
    configPrettier,

    {
        files: ['**/*.js'],
        languageOptions: {
            ecmaVersion: 'latest',
            sourceType: 'commonjs',
            globals: {
                ...globals.browser,
                ...globals.node,
            },
        },
        plugins: {
            prettier,
        },
        rules: {
            'no-unused-vars': [
                'error',
                {
                    argsIgnorePattern: '^_',
                    varsIgnorePattern: '^_',
                    caughtErrorsIgnorePattern: '^_',
                },
            ],
            'prettier/prettier': [
                'error',
                {
                    tabWidth: 4,
                    useTabs: false,
                    singleQuote: true,
                    semi: true,
                    printWidth: 100,
                    trailingComma: 'es5',
                },
            ],
        },
    },

    {
        files: ['demo.js'],
        languageOptions: {
            globals: {
                hpspline: 'readonly',
            },
        },
    },

    {
        ignores: ['node_modules/'],
    },
];
