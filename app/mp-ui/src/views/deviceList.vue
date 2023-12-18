<template>
    <div>
        <search
            v-model="tree_module.query.name"
            placeholder="设备名"
            @search="onQuery"
        />
        <van-pull-refresh v-model="tree_module.statue.loading" @refresh="onRefresh">
            <card
                v-for="item in tree_module.data"
                :key="item.id"
                :title="item.name"
                :thumb="getPoster(item.id)"
            >
                <template #tags>
                    <van-tag plain type="danger">{{ item.ip }}</van-tag>
                    <van-tag plain type="danger">{{ item.innerIp }}</van-tag>
                </template>
                <template #footer>
                    <van-button size="mini" @click="onPreview(item)"
                        >预览</van-button
                    >
                </template>
            </card>
        </van-pull-refresh>
        <!--
        <divider /> 
        <van-image
            width="100"
            height="100"
            fit="contain"
            src="https://fastly.jsdelivr.net/npm/@vant/assets/cat.jpeg"
        />-->
    </div>
</template>
<script setup lang="ts">
import { ref, reactive } from "vue";
import {
    Image as VanImage,
    PullRefresh as vanPullRefresh,
    Divider,
    List,
    Search,
    Icon,
    Grid,
    GridItem,
    Card,
    Tag as vanTag
} from "vant";

import * as api from "../api/device";
import * as res_api from "../api";
import * as d from "../store/types/devices";

import { useAppDataStore } from "../store/appStore";
import { useRouter } from "vue-router";
import { showNotify } from 'vant';
const router = useRouter();
const dataStore = useAppDataStore();

interface Tree {
    [key: string]: any;
}
interface Query extends IpageParam {
    name: string;
}

interface tree_module {
    query: Query;
    data: Array<d.dataItem>;
    currentItem?: d.dataItem;
    total: number;
    defaultProps: { children: String; label: String };
    filterNode: (value: string, data: Tree) => boolean;
    statue: {
        loading: boolean;
        finished: boolean;
        refreshing: boolean;
    };
} 
const tree_module = reactive<tree_module>({
    query: {
        name: "",
        pageIndex: 1,
        pageSize: 10
    },
    statue: {
        loading: false,
        finished: false,
        refreshing: false
    },
    data: [],
    total: -1,
    filterNode: (value: string, data: Tree) => {
        if (!value) return true;
        return data.label.includes(value);
    },
    defaultProps: {
        children: "children",
        label: "name"
    }
});
const getData = () => {
    tree_module.statue.loading=true
    api.list_svc(tree_module.query).then((res) => {
        if (res.code == 0) {
            tree_module.data.push(...res.data);
            tree_module.total = res.total || -1;
        } else showNotify({type:"danger",message:res.message})
        if (tree_module.data.length >= res.total) {
            tree_module.statue. finished = true;
        }
        if (res.data.length > 0) { 
            tree_module.query.pageIndex++;
        } 
        tree_module.statue.loading=false
    });
};
const onQuery = () => {
    tree_module.data = [];
    getData();
};
const onRefresh = () => {
    // 清空列表数据
    tree_module.statue.finished = false; 
    tree_module.data = [];
    
    getData();
};
getData();

const getPoster = (id: number) => {
    return "https://fastly.jsdelivr.net/npm/@vant/assets/cat.jpeg"; //await api.get_poster_svc(id)
};

const onPreview = (row: d.dataItem) => {
    dataStore.setState(row);
    router.push({
        path: "/preview.html"
    });
};
</script>
