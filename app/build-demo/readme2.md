1. npm i -D typescript # 目录下多了 node_modules 文件夹和 package-lock.json 文件。

2. 为了编译 TypeScript 创建 tsconfig.json

```
{
	"compilerOptions": {
		"target": "es5", //编译之后生成的 JavaScript 文件需要遵循的标准,"ES3"（默认），"ES5"，"ES6"/"ES2015"，"ES2016"，"ES2017"或"ESNext"
		"module": "commonjs", //指定生成哪个模块系统代码 "ES3" or "ES5" ? "CommonJS" : "ES6"
		"declaration": true,//是否生成对应的声明文件 以便包可以在 TypeScript 和 JavaScript 项目中同时使用。
		"outDir": "./lib",
		"strict": true, //是否启用所有严格类型检查选项
		"lib": ["es6"]
	},
	"include": ["src"],
	"exclude": ["node_modules", "**/__tests__/*"]
}

```

3. 创建index 文件
4. 在 package.json 中添加 build script

```
"build": "tsc"
```

5. 运行’npm run build‘
6. .gitignore

```
node_modules
/lib
```

7. Formatting & Linting
   开发阶段所需的工具,tslint-config-prettier 能防止 TSLint 和 Prettier 格式化规则的冲突。

```
npm i -D prettier tslint tslint-config-prettier
```

## 7.1 tslint.json

```
{
  "extends": ["tslint:recommended", "tslint-config-prettier"]
}
```

## 7.2. .prettierrc

```
{
  "printWidth": 120,
  "trailingComma": "all",
  "singleQuote": true
}
```

## 7.3 package.json .scripts 中添加 lint 和 format script

```
"format": "prettier --write \"src/**/*.ts\" \"src/**/*.js\"",
"lint": "tslint -p tsconfig.json"
```

# 8. 发布
- 黑名单 `.npmignore`
```
src
tsconfig.json
tslint.json
.prettierrc
```
- 白名单  package.json.`files:[]` 

# 9. 单元测试

Jest —— Facebook 开发的一个非常棒的测试框架
ts-jest @types/jest 针对 ts 源文件编写测试用例

```
npm i -D jest ts-jest @types/jest
```

## 9.1 配置jest

- package.json 添加 "jest" 字段
- 新建 jestconfig.json

```
{
    "transform": {
        "^.+\\.(t|j)sx?$": "ts-jest"
    },
    "testRegex": "(/__tests__/.*|(\\.|/)(test|spec))\\.(jsx?|tsx?)$",
    "moduleFileExtensions": [
        "ts",
        "tsx",
        "js",
        "jsx",
        "json",
        "node"
    ]
} 
``` 
- 在pageage.json 中增加

```
"test": "jest --config jestconfig.json",
```

## 9.2 测试用例

src 中 新建一个 **tests** 文件夹，文件名必须以 test.ts 结尾

```
import { Greeter } from '../index';
test('My Greeter', () => {
  expect(Greeter('Carl')).toBe('Hello Carl');
});
```

## 9.3 运行

```
npm run test
```

# 10. 优化

好的包应该尽可能自动化，prepare，prepublishOnly，perversion，version，postversion

- prepare：会在打包和发布包之前以及本地 npm install （不带任何参数）时运行。

```
"prepare": "npm run build"
```

- prepublishOnly：在 prepare script 之前运行，并且仅在 npm publish 运行。在这里，我们可以运行 npm run test & npm run lint 以确保我们不会发布错误的不规范的代码。

```
"prepublishOnly": "npm run test && npm run lint"
```

- preversion 在发布新版本包之前运行，为了更加确保新版本包的代码规范，我们可以在此运行 npm run lint

```
"preversion": "npm run lint"
```

- version 在发布新版本包之后运行。如果您的包有关联远程 Git 仓库,每次发布新版本时都会生成一个提交和一个新的版本标记，那么就可以在此添加规范代码的命令。又因为 version script 在 git commit 之前运行，所以还可以在此添加 git add

```
"version": "npm run format && git add -A src"
```

- postversion 在发布新版本包之后运行，在 git commit之后运行，所以非常适合推送

```
"postversion": "git push && git push --tags"
```
