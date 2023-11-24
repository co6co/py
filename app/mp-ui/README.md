 
## 安装步骤
> 因为使用vite3，node版本需要 14.18+

```
git clone https://github.com/lin-xin/vue-manage-system.git      // 把模板下载到本地
cd vue-manage-system    // 进入模板目录
npm install         // 安装项目依赖，等待安装完成之后，安装失败可用 cnpm 或 yarn

// 运行
npm run dev

// 执行构建命令，生成的dist文件夹放在服务器下即可访问
npm run build
```

ERROR 处理方式:
Top-level await is not available in the configured target environment ("chrome87", "edge88", "es2020", "firefox78", "safari13" + 2 overrides)
>npm install vite-plugin-top-level-await -D
 
 # 调试
  vue-devtools 5.3.4 及以下为 vue2