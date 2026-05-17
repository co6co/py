1. Terser 代码 混编
 去掉 console.log 等
 比如:vue-cli搭建的工程:

 ```
 // vlie 内置了 terger
 module.exports=defineConfig{
    terger:{
        terserOptons:{
            compress:{
                drop_console:true,
                drop_debugger:true
            }
        }
    }
 }

 vite:
 需要单独安装
 build:'terser':
 {
    minify:'terser', // 默认方式不是terser
    terserOption:{
        compress:{
            drop_console:true,
            drop_debugger:true
        }
    }
 }
 ```