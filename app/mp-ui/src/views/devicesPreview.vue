<template>
    <div>
        <div class="container">
            <div class="header">
                <div class="collapse-btn" @click="onSideBarBut">
                    <el-icon v-if="sidebar.collapse"><Expand /></el-icon>
                    <el-icon v-else><Fold /></el-icon>
                </div>
            </div>
            <el-row :gutter="24">
                <el-col :span="colsWidth.l">
                    <el-input v-model="deviceName" placeholder="设备名称" />
                    <el-tree
                        @node-click="onNodeCheck"
                        ref="tree"
                        class="filter-tree"
                        :data="tree_module.data"
                        :props="tree_module.defaultProps"
                        default-expand-all
                        :filter-node-method="tree_module.filterNode"
                    />
                </el-col>
                <el-col :span="colsWidth.c">
                    <stream :sources="player.sources"></stream>
                </el-col>
                <el-col :span="colsWidth.r"><ptz @ptz="OnPtz"></ptz></el-col>
            </el-row>
        </div>
    </div>
</template>
<script setup lang="ts" name="basetable">
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
    Download
} from "@element-plus/icons-vue";
import * as api from "../api/device";
import { stream, ptz } from "../components/stream";
import { useMqtt } from "../utils/mqtting";

import { useSidebarStore } from "../store/sidebar";
const sidebar = useSidebarStore();

interface ColsWidth {
    l: 3 | 1 | 0;
    c: 15 | 22 | 24;
    r: 6 | 1 | 0;
}
const colsWidth = reactive<ColsWidth>({
    l: 3,
    c: 15,
    r: 6
});
const onSideBarBut = () => {
    if (sidebar.collapse) {
        colsWidth.l = 0;
        colsWidth.c = 24;
        colsWidth.r = 0;
    } else {
        colsWidth.l = 3;
        colsWidth.c = 15;
        colsWidth.r = 6;
    }
    sidebar.handleCollapse();
};
const deviceName = ref("");
const tree = ref(null);

interface Tree {
    [key: string]: any;
}
interface Query {
    name: string;
}
interface dataItem {}
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
// 查询操作
const onSearch = () => {
    getData();
};
const getQuery = () => {};
// 获取表格数据
const getData = () => {
    getQuery(),
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
/** 播放器 */
interface player_sources {
    sources: Array<stream_source>;
}
const player = reactive<player_sources>({ sources: [] });

const onNodeCheck = (item?: any) => {
    tree_module.currentItem = item;
    player.sources = [
           
	  {url:`http://wx.co6co.top:452/flv/vlive/${item.ip}.flv`,name:"HTTP-FLV"}, 
		{url:`ws://wx.co6co.top:452/ws-flv/vlive/${item.ip}.flv`,name:"WS-FLV"}, 
		{url:`webrtc://wx.co6co.top:452/rtc/vlive/${item.ip}`,name:"webrtc"}, 
		{url:`http://wx.co6co.top:452/vhls/${item.ip}/${item.ip}_live.m3u8`,name:"HLS(m3u8)"}
    
    ];
};
/** ptz */
const { startMqtt, Ref_Mqtt } = useMqtt();
interface mqttMessage {
    UUID?: string;
}
let arr: Array<mqttMessage> = [];
startMqtt(
    "WS://192.168.1.99:4451/mqtt",
    "/edge_app_controller_reply",
    (topic: any, message: any) => {
        const msg: mqttMessage = JSON.parse(message.toString());
        arr.unshift(msg); //新增到数组起始位置
        console.warn(unique(arr));
    }
);

function unique(arr: Array<mqttMessage>) {
    const res = new Map();
    return arr.filter((a) => !res.has(a.UUID) && res.set(a.UUID, 1));
}
const OnPtz = (name: string, type: string) => {
    let param = {
        payload: {
            BoardId: "RJ-BOX3-733E5155B1FBB3C3BB9EFC86EDDACA60",
            Event: "/app_network_query_v2"
        },
        qos: 0,
        retain: false
    };
    Ref_Mqtt.value?.publish(
        "/edge_app_controller",
        JSON.stringify(param.payload)
    );
    console.warn(name, type);
};
//**end 打标签 */
</script>
<style lang="less">
.el-link {
    margin-right: 8px;
}
.el-link .el-icon--right.el-icon {
    vertical-align: text-bottom;
}
.handle-box {
    margin: 3px 0;
}
.handle-select {
    width: 120px;
}

.handle-input {
    width: 300px;
}
.table {
    width: 100%;
    font-size: 14px;
}
.red {
    color: #f56c6c;
}
.mr10 {
    margin-right: 10px;
}
.table-td-thumb {
    display: block;
    margin: auto;
    width: 40px;
    height: 40px;
}
.view .title {
    color: var(--el-text-color-regular);
    font-size: 18px;
    margin: 10px 0;
}
.view .value {
    color: var(--el-text-color-primary);
    font-size: 16px;
    margin: 10px 0;
}

::v-deep .view .radius {
    height: 40px;
    width: 70%;
    border: 1px solid var(--el-border-color);
    border-radius: 0;
    margin-top: 20px;
}
::v-deep .el-table tr,
.el-table__row {
    cursor: pointer;
}

.formItem {
    display: flex;
    align-items: center;
    display: inline-block;
    .label {
        display: inline-block;
        color: #aaa;
        padding: 0 5px;
    }
}

::v-deep .el-dialog__body {
    height: 70%;
    overflow: auto;
}
.menuInfo {
    .el-menu {
        width: auto;
        .el-menu-item {
            padding: 10px;
            height: 40px;
        }
    }
}

/**
  
  ::v-deep .el-dialog{
      .el-dialog__header{padding: 5px;}
      .el-dialog__body{padding: 15px 5px;}
      .el-dialog__footer{padding: 5px;}
  } */
.content {
    padding: 0;
}
.container {
    padding: 0;
}
.header {
    padding: 8px;
    font-size: 28px;
}
.collapse-btn {
    color: white;
}
</style>
