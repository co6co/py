<template>
    <div v-loading="loading">  </div>
</template>
<script setup lang="ts">
import { ref, reactive } from "vue";
import { randomString } from "../utils";
import { getRedirectUrl } from "../components/wx" 
const backUrk=ref()
const redirect_uri = import.meta.env.VITE_WX_redirect_uri; 
const scope=1
//let redirectUrl =getRedirectUrl(redirect_uri,scope,`${randomString(10)}-${scope}-/mp/ui/-${randomString(10)}`)
let redirectUrl =getRedirectUrl(redirect_uri,scope,`${randomString(10)}-${scope}-http://127.0.0.1:5174/**/usermgr-${randomString(10)}`)
window.location.href = redirectUrl;
backUrk.value=redirect_uri
const loading = ref(true)
</script>
<style lang="less">
body {
  margin: 0;
}
.example-showcase .el-loading-mask {
  z-index: 9;
}
</style>
