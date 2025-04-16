import { basicSetup } from 'codemirror';
import { python } from '@codemirror/lang-python';
import { javascript } from '@codemirror/lang-javascript';
import { oneDarkTheme } from '@codemirror/theme-one-dark';
import { EditorView, keymap } from '@codemirror/view';
import { tags } from '@lezer/highlight';
import { HighlightStyle } from '@codemirror/language';
import { syntaxHighlighting } from '@codemirror/language';
import { indentWithTab } from '@codemirror/commands';
let myTheme = EditorView.theme(
	{
		'&': {
			color: 'white',
			backgroundColor: '#034',
		},
		'.cm-content': {
			caretColor: '#0e9',
		},
		'&.cm-focused .cm-cursor': {
			borderLeftColor: '#0e9',
		},
		'&.cm-focused .cm-selectionBackground, ::selection': {
			backgroundColor: '#fff',
		},
		'.cm-gutters': {
			backgroundColor: '#045',
			color: '#ddd',
			border: 'none',
		},
	},
	{ dark: true }
);
const myHighlightStyle = HighlightStyle.define([
	{ tag: tags.keyword, color: '#fc6' },
	{ tag: tags.comment, color: '#f5d', fontStyle: 'italic' },
]);
const cmOptions = {
	theme: oneDarkTheme,
	extensions: [
		basicSetup,
		python(),
		javascript(),
		myTheme,
		syntaxHighlighting(myHighlightStyle),
		keymap.of([indentWithTab]),
	],
};
export default cmOptions;
