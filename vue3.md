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
