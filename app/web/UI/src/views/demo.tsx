import { defineComponent, onMounted, ref, VNode } from 'vue'

import CodeMirror from 'vue-codemirror6'
import { basicSetup, minimalSetup } from 'codemirror'

import { python } from '@codemirror/lang-python'
import { javascript } from '@codemirror/lang-javascript'
import { oneDark } from '@codemirror/theme-one-dark'
import { EditorView, keymap, lineNumbers } from '@codemirror/view'
import { tags } from '@lezer/highlight'
import { HighlightStyle } from '@codemirror/language'
import { syntaxHighlighting } from '@codemirror/language'
import { indentWithTab } from '@codemirror/commands'

export default defineComponent({
  setup(prop, ctx) {
    //存储本地值
    const localValue = ref(`
<html lang="en">
  <head>
    <script type="module" src="/system/@vite/client"></script>

    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/system/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title></title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/system/src/main.ts?t=1729236056017"></script> 
  </body>
</html>
      `)
    let myTheme = EditorView.theme(
      {
        '&': {
          color: 'white',
          backgroundColor: '#034'
        },
        '.cm-content': {
          caretColor: '#0e9'
        },
        '&.cm-focused .cm-cursor': {
          borderLeftColor: '#0e9'
        },
        '&.cm-focused .cm-selectionBackground, ::selection': {
          backgroundColor: '#074'
        },
        '.cm-gutters': {
          backgroundColor: '#045',
          color: '#ddd',
          border: 'none'
        }
      },
      { dark: true }
    )
    const myHighlightStyle = HighlightStyle.define([
      { tag: tags.keyword, color: '#fc6' },
      { tag: tags.comment, color: '#f5d', fontStyle: 'italic' }
    ])
    const cmOptions = {
      theme: oneDark,
      extensions: [
        python(),
        //javascript(),
        myTheme,
        lineNumbers(),
        keymap.of([indentWithTab])
      ]
    }
    const codeMirrorRef = ref()
    onMounted(() => {
      console.info(codeMirrorRef.value)
    })
    const rander = (): VNode => {
      return (
        <div style="padding:5px">
          <CodeMirror
            ref={codeMirrorRef}
            style="width:99%; border:1px solid #ccc"
            v-model={localValue.value}
            dark
            basic
            tab
            tabSize={4}
            lang={python()}
            gutter
            extensions={cmOptions.extensions}
          />
        </div>
      )
    }
    return rander
  }
})
