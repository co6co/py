<template>
    <div v-loading="loading">
        <div>
            <v-tags></v-tags>
            <div class="content">
                <router-view v-slot="{ Component }">
                    <transition name="move" mode="out-in">
                        <keep-alive :include="tags.nameList">
                            <component :is="Component"></component>
                        </keep-alive>
                    </transition>
                </router-view>
            </div>
        </div>
    </div>
</template>
<script setup lang="ts">
import { ref, reactive, onMounted } from "vue";
import { randomString } from "../utils";
import { getRedirectUrl } from "../components/wx";

import { useTagsStore } from "../store/tags";
import vTags from "../components/tags.vue";
import { getTokes, setTokes } from "../utils/auth";
import { ticket_svc } from "../api/user";
import { nextTick } from "process";
const tags = useTagsStore();

const getUrl = () => {
    let url = document.location.toString();
    let arrUrl = url.split("//");
    let start = arrUrl[1].indexOf("/");
    let relUrl = arrUrl[1].substring(start); //stop省略，截取从start开始到结尾的所有字符
    if (relUrl.indexOf("?") != -1) {
        relUrl = relUrl.split("?")[0];
    }
    return relUrl.replace("#", "**");
};

function getQueryVariable(key: string) {
    try {
        //var query = window.location.search.substring(1);
        var query = window.location.href.substring(
            window.location.href.indexOf("?") + 1
        );
        var vars = query.split("&");
        for (var i = 0; i < vars.length; i++) {
            var pair = vars[i].split("=");
            if (pair[0] == key) {
                return pair[1];
            }
        }
        return null;
    } catch (e) {}
    return null;
}
let token = getTokes();
const ticket = getQueryVariable("ticket");


if (!token && ticket) {
     ticket_svc(ticket).then((res)=>{
        if (res.code == 0) setTokes(res.data), token = getTokes();
    }) 
}  
if (!token) {
    const backUrk = ref();
    const redirect_uri = import.meta.env.VITE_WX_redirect_uri;
    const scope = 1;
    let redirectUrl = getRedirectUrl(
        redirect_uri,
        scope,
        `${randomString(10)}-${scope}-${getUrl()}-${randomString(10)}`
    );
     /*
        redirectUrl = getRedirectUrl(
            redirect_uri,
            scope,
            `${randomString(10)}-${scope}-${getUrl()}-${randomString(10)}`
        );
        */
 
    window.location.href = redirectUrl;
}

const loading = ref(true);
onMounted(() => {
    loading.value = false;
});
</script>
<style lang="less">
body {
    margin: 0;
}
.example-showcase .el-loading-mask {
    z-index: 9;
}
</style>
