<template>
    <div>
        获取用snsapi_userinfo
        {{ backUrk }}
        <br/>
        {{ redirectUrl }}

    </div>
</template>
<script setup lang="ts">
import { ref, reactive } from "vue";
import { randomString } from "../utils";
import { getRedirectUrl } from "../components/wx" 
const backUrk=ref()
const redirect_uri = import.meta.env.VITE_WX_redirect_uri; 
const scope=1
let redirectUrl =getRedirectUrl(redirect_uri,scope,`${randomString(10)}-${scope}-/mp/ui/-${randomString(10)}`)
window.location.href = redirectUrl;
backUrk.value=redirect_uri
</script>
<style lang="less"></style>
