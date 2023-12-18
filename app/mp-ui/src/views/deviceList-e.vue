<template>
    <div>
        <ul style="height: 5rem">
            <li v-for="item in tree_module.data">
                <el-row :gutter="24">
                    <el-col :span="8">
                        <el-image
                            :src="item.poster"
                            style="
                                width: 80px;
                                height: 60px;
                                display: inline-block;
                            "
                        >
                            <template #error>
                                <div class="image-slot">
                                    <el-icon><icon-picture /></el-icon>
                                </div> </template
                        ></el-image>
                    </el-col>
                    <el-col :span="16">
                        <el-row>
                            <span>{{ item.name }}</span>
                        </el-row>
                        <el-row>
                            <span>
                                <el-icon><VideoCameraFilled /></el-icon>
                            </span>
                        </el-row>
                    </el-col>
                </el-row>
            </li>
        </ul>

        <el-pagination
            background
            layout="prev, pager, next"
            :total="tree_module.total"
        />
    </div>
</template>

<script setup lang="ts">
import {
    ref,
    watch,
    reactive,
    watchEffect,
    nextTick,
    PropType,
    onMounted,
    onBeforeUnmount,
    computed
} from "vue";
import {
    ElMessage,
    ElMessageBox,
    FormRules,
    FormInstance,
    ElTreeSelect
} from "element-plus";
import { TreeNode } from "element-plus/es/components/tree-v2/src/types";
import { TreeNodeData } from "element-plus/es/components/tree/src/tree.type";
import {
    Delete,
    Edit,
    Search,
    Compass,
    MoreFilled,
    Download,
    VideoCameraFilled,
    Picture as IconPicture
} from "@element-plus/icons-vue";
import * as api from "../api/device";

interface Tree {
    [key: string]: any;
}
interface Query {
    name: string;
}
interface dataItem {
    name: string;
    poster: string;
}
interface tree_module {
    query: Query;
    data: Array<dataItem>;
    currentItem?: dataItem;
    total: number;
    defaultProps: { children: String; label: String };
    filterNode: (value: string, data: Tree) => boolean;
}

const tree_module = reactive<tree_module>({
    query: {
        name: ""
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
    api.list_svc(tree_module.query).then((res) => {
        if (res.code == 0) {
            tree_module.data = res.data;
            tree_module.total = res.total || -1;
        } else {
            ElMessage.error(res.message);
        }
    });
};
getData();
</script>
<style lang="less">
.el-image {
    padding: 0 5px;
    max-width: 300px;
    max-height: 200px;
    width: 100%;
    height: 200px;
}

ul {
    height: 300px;
    padding: 0;
    margin: 0;
    list-style: none;
    li {
        cursor: pointer;
        display: flex;
        align-items: left;
        justify-content: left;

        background: var(--el-color-primary-light-9);
        margin: 10px;
        color: var(--el-color-primary);
        .image-slot {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            height: 100%;
            background: var(--el-fill-color-light);
            color: var(--el-text-color-secondary);
            font-size: 30px;
        }
        span {
            display: inline-block;
            padding: 5px;
            margin-left: 15px;
        }
    }
}
</style>
