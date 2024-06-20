# 1. 文档
```
{
  "name": "co6co_permission", //npm install  xxx  
  "version": "0.0.0.1", ////组件库的版本
  "main": "./src/index.ts", ////上传库的入口文件（重要）
  "private": false, //true私有的，在npm上上传组件库为私有的需要付费，所以设置为false则为公共
  "auther":"co6co",
  "license": "ISC",
  "type": "module",
  "scripts": {
    "dev": "vite --mode dev",
    "preview": "vite preview --port 8080",
    "build": "vite build --mode pro", 
    "lib": "vite build --target lib --name co6co_permission"  
  },
  "dependencies": {
    "@element-plus/icons-vue": "^2.3.1",
    "axios": "^1.6.7",
    "element-plus": "^2.5.3",
    "js-cookie": "^3.0.5",
    "json-bigint": "^1.0.0",
    "less": "^4.2.0",
    "pinia": "^2.1.7",
    "vue": "^3.3.11",
    "vue-cropperjs": "^5.0.0",
    "vue-router": "^4.2.5",
    "wangeditor": "^4.7.15",
    "xlsx": "https://cdn.sheetjs.com/xlsx-0.20.0/xlsx-0.20.0.tgz"
  },
  "devDependencies": {
    "@rushstack/eslint-patch": "^1.3.3",
    "@tsconfig/node18": "^18.2.2",
    "@types/js-cookie": "^3.0.6",
    "@types/jsdom": "^21.1.6",
    "@types/node": "^18.19.3",
    "@vitejs/plugin-vue": "^4.5.2",
    "@vitejs/plugin-vue-jsx": "^3.1.0",
    "@vue/eslint-config-prettier": "^8.0.0",
    "@vue/eslint-config-typescript": "^12.0.0",
    "@vue/test-utils": "^2.4.3",
    "@vue/tsconfig": "^0.5.0",
    "eslint": "^8.49.0",
    "eslint-plugin-vue": "^9.17.0",
    "jsdom": "^23.0.1",
    "npm-run-all2": "^6.1.1",
    "prettier": "^3.0.3",
    "typescript": "~5.3.0",
    "typescript-plugin-css-modules": "^5.0.2",
    "unplugin-auto-import": "^0.17.5",
    "unplugin-vue-components": "^0.26.0",
    "vite": "^5.0.10",
    "vite-plugin-top-level-await": "^1.4.1",
    "vitest": "^1.0.4",
    "vue-tsc": "^1.8.25"
  }
}
```
```
"lib": "set NODE_OPTIONS=--openssl-legacy-provider && vue-cli-service build --target lib --name cjuly-ui --dest lib ./src/components/index.js"

```
# 2. `npm run dev` //跑一下，确定没问题再上传。
# 3. NPM 发布包
## 3.1 登录
```
npm login 
npm who am i # 查看是否登录

```
## 3.2 发布
打包前忽略的文件，源文件一般在 lib 或 src 目录下，但生成的文件一般在 dist 或 build 目录下，
而我们会把这个构建目录在 .gitignore 中给忽略掉，导致发包之后也看不到。
1. 增加 `.npmignore `文件，把忽略的文件写在这里，因为默认情况下，npm 打包会根据 `.gitignore `来忽略文件，如果发现存在 `.npmignore` 的话，则使用这个文件。
2. 在 `package.json` 中增加 files 数组，把需要打包的文件写在这里，那么 npm 打包的时候只会按照你列出来的规则添加文件。它的优先级高于 `.npmignore` 和 `.gitignore`，且支持通配符。
```
常用发布方法：
只想把 dist 目录生成的文件打包上传，并不想提供源代码，或者我们是用 TypeScript 写的，发包的时候只发编译过的 JavaScript 文件和 d.ts 声明文件。此时只需要把 package.json 拷贝一份到 dist 目录，然后在该目录下运行 npm publish 即可。

为了防止搞混且误提交，可以在源代码中的 package.json 添加 private: true 字段，而在 dist 的 package.json 中去掉该字段。
 
//切换到npm地址
//可能使用到的命令
npm config set registry=https://registry.npmjs.org
npm config edit #打开文本文件编辑
 
registry=https://registry.npmmirror.com

# registry 切换 
# nrm 为 registry 切换工具
npm install -g nrm 
nrm use taobao
nrm use npm

# 发布前 检查那些文件会被发布出去 或者本地打个包看看有那些文件
npm publish --dry-run # 查看有文件
npm pack              # 打个包看看有那些文件

//因为上传私有组件需要付费，所以上传为公共的
npm publish --access public 


```
## 3.3 其他相关操作

```
# 查看远程是否存在包名
npm view 包名

# 配置初始化
npm init
npm init -y   # ts 项目使用  tsc --init
# 删除
npm unpublish dzmtest@1.0.1 --force
npm unpublish dzmtest --force
npm unpublish --force # 删除 package.json 的包名相应的版本

# 废弃 安装时会有警示，并不影响使用。
npm deprecate <pkg>[@<version>] <message> 

```
# 4. 使用
```
npm install xxx
```
# 5. 其他相关
```

添加 "declaration": true 到你的 tsconfig.json --> 告诉 TypsScript 在编译的时候为你自动生成 d.ts 文
添加 "types": "index.d.ts" 到你的 package.json --> 导入包时，告诉了 TS 编译器到哪里去寻找类型定义文件

```
