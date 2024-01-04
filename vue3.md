# 使用淘宝镜像
```
 npm install -g cnpm --registry https://registry.npm.taobao.org
 npm install cnpm -g --registry=https://registry.npmmirror.com
 
 npm config set registry https://registry.npmjs.org
```

# 导入资源
@ 把它当做src文件夹的别名
~@ 可以加载静态资源(图片与css),也可以加载node-modules中的资源,不可以加载JavaScript与typescript
```
<template>
<div>
  <img src="~@/assets/image/PE.png">     ~@ 可以
  <img src="@/assets/image/PE.png">       @  可以
  <img src="../assets/image/PE.png">     相对路径可以
  <img src="/image/admin.png"">          public绝对路径可以
  <div class="adminsvg"></div>
</div>
</template>

<script>
// import("~@/assets/iconfont/iconfont")              ~@ 不可以
import("@/assets/iconfont/iconfont.js")                @  可以
import("../assets/iconfont/iconfont")                  相对路径可以
// import("/iconfont/iconfont")                        public绝对路径不可以
</script>

<style scoped>                
@import url("~@/assets/css/register_style.css");        ~@ 可以
@import url("@/assets/css/register_style.css");         @ 可以
@import url("../assets/css/register_style.css");        相对路径可以
// @import url("/css/register_style.css");              public绝对路径不可以

.adminsvg {
    background-image: url("~@/assets/image/PE.png");     ~@ 可以
    background-image: url("@/assets/image/PE.png");       @ 可以
    background-image: url("../assets/image/PE.png");     相对路径可以
    // background-image: url("/image/PE.png");           public绝对路径不可以
}
</style> 
```


# 对象
https://ts.xcatliu.com/basics/type-of-object-interfaces.html
https://blog.csdn.net/wenxingchen/article/details/125341773 动态组件

<<<<<<< HEAD
https://cn.vuejs.org/guide/quick-start.html
=======
# UI 
https://lyt-top.gitee.io/vue-next-admin-preview/#/system/role


https://hansuku.com/blog/0031
