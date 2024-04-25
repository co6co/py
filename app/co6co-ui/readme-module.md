# 1. 生成模块代码
”target” 用于指定 TypeScript 编译器的目标版本，”module” 用于指定模块生成方式。
 
- target   设置为所要支持的最低版本。ES3[默认]|ES5|ES2015|ES2016
- module   指定 TypeScript 编译后的 JavaScript 代码的模块生成方式 none|commonjs[默认]|amd|system|umd|es2015 
实例：
1. 当target为es5时，如果没有指定module选项 则默认值为commonjs
没有指定moduleResolution选项时，默认值是node
这种情况下编译出来的模块是commonjs方式的，也就是node环境下执行的代码
模块导入方式也是node方式导入的。

2. 当target为es6时，module选项的值会默认为ES6，moduleResolution选项的值默认为Classic，
则此时导入包的方式就会找不到包，
所以需要手动指定moduleResolution选项为node，如果代码执行环境为node还需要指定module为commonjs。

|参数|参数|语句|说明|
|--|--|--|--|
|--module 或者 -m |string|target =="ES6"?"ES6":"commonjs"|指定生成那个模块系统：None/CommonJS/AMD/SYSTEM/UDM/ES6/ES2015，只有AMD和System能和--outfile一起使用，ES6和ES2015可以使用目标为ES5或更低的情况|
|--moduleResolution |string|module=="AMD" or "SYSTEM" or ES6?"Classic":"Node"|决定如何处理模块： Node或者 Classic|


# ES5 和ES6
- ES5 ECMScript的第五个版本，发布于2009年，是目前最广泛使用的JavaScript版本。
- ES6 ECMScript的第六个版本，也成为ES2015，发布于2015年，引入了许多新的语言特性和语法糖。 ES2015是ES6的官方名称，但是由于ES6引入了太多的新特性，因此人们通常使用ES2015来指代ES6。 