

<template>
    <div>
        home
    </div>
</template>
<script setup lang="ts">
import {ref ,reactive} from 'vue'
import {randomString} from "../utils"
const appid=import.meta.env.VITE_WX_appid
const redirect_uri=import.meta.env.VITE_WX_redirect_uri
let stateCode = randomString(18);
let redirectUrl = 'https://open.weixin.qq.com/connect/oauth2/authorize?appid=' + appid; 
redirectUrl += '&redirect_uri=' +redirect_uri
redirectUrl += '&response_type=code&scope=snsapi_userinfo&state=' + stateCode + '#wechat_redirect'
window.location.href = redirectUrl 
</script>
<style lang="less">

</style>
