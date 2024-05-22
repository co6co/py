# 1. 小知识
## 1.1. 更改package.json 中的版本号
```
npm version 0.0.2
``` 
## 1.1.1 __dirname| path
```
来自 @types/node 包
```


## 1.2. 发布时打标签
```
//发布 为发布测试版使用
npm publish --tag beta
// 安装使用
npm install my-package@beta

npm install my-package == npm install my-package@latest
```


## 1.3. 发布前package.json 配置
```
name: 包名，该名字是唯一的。可在 npm 官网搜索名字，如果存在则需换个名字。
version: 版本号，不能和历史版本号相同。
files: 配置需要发布的文件。
main: 入口文件，默认为 index.js，这里改为 dist/vue3-xmw-table.umd.js。
module: 模块入口，这里改为 dist/vue3-xmw-table.es.js。
```

# 2 jsx 
## 2.1  dom元素全部都显示 找不到名称“react”
tsconfig.json配置文件的include 路径问题
## 2.2 JSX 元素隐式具有类型 "any"，因为不存在接口 "JSX.IntrinsicElements"。
```
//tsconfig.json
{
  "compilerOptions": {
    "jsx": "preserve",
    "jsxImportSource": "vue"
    // ...
  }
}
如果一直不生效，重启下vscode
```

## 2.3 找不到 *.less 模块
```
declare module '*.scss' {
  const classes: { [key: string]: string };
  export default classes;
}
declare module '*.less' {
  const classes: { [key: string]: string };
  export default classes;
}
```
或者引入定义：
```
/// <reference types="vite/client" />
```