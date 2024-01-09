<template>
    <el-card class="box-card">
        <!--header-->
        <template #header>
            <div class="card-header">
                <el-input v-model="tree_module.query.name" placeholder="点位名称">
                    <template #append>
                        <el-button :icon="Search" @click="tree_module.onSearch" />
                    </template>
                </el-input>
            </div>
        </template>
        <!--content-->
        <el-scrollbar>
            <div class="content">
                <el-tree v-if="hasData" highlight-current @node-click="onNodeCheck" ref="tree" class="filter-tree"
                    :data="tree_module.data" :props="tree_module.defaultProps" default-expand-all
                    :filter-node-method="tree_module.filterNode">
                    <template #default="{ node, data }">
                        <span>
                            <!-- 没有子级所展示的图标 -->
                            <i v-if="!data.devices"><el-icon>
                                    <VideoCamera />
                                </el-icon></i>
                            <i v-else-if="data.devices"><el-icon>
                                    <Avatar />
                                </el-icon></i>
                            <span class="label">
                                <el-tooltip :content="node.label">
                                    {{ node.label }}
                                </el-tooltip>
                            </span>
                        </span>
                    </template>
                </el-tree>
                <el-empty v-else></el-empty>
            </div>
        </el-scrollbar>
        <!--footer-->
        <template #footer>
            <div class="context">
                <el-pagination v-if="hasData" background layout="prev,next" :total="tree_module.total"
                    :current-page="tree_module.query.pageIndex" :page-size="tree_module.query.pageSize"
                    @current-change="tree_module.pageChange" />
            </div>
        </template>
    </el-card>
</template>
<script setup lang="ts">
import {
  ref,
  reactive,
  computed,
} from 'vue';
import {
  ElMessage,
  ElMessageBox,
  FormRules,
  FormInstance,
  ElTreeSelect,
} from 'element-plus';
import { TreeNode } from 'element-plus/es/components/tree-v2/src/types';
import { TreeNodeData } from 'element-plus/es/components/tree/src/tree.type';
import {
  Delete,
  Edit,
  Search,
  Compass,
  MoreFilled,
  Download,
  CloseBold,
  VideoCamera,
  Avatar,
  ArrowUp,
  ArrowDown,
} from '@element-plus/icons-vue';
import * as api from '../../..//api/site';
import { showLoading, closeLoading } from '../../../components/Logining';
import * as types from "./types";

interface Emits { (e: 'nodeClick', streams: String | { url: string; name: string }, device: types.deviceItem): void; }
const emits = defineEmits<Emits>();
const tree = ref(null);
interface Tree {
  [key: string]: any;
}
interface Query extends IpageParam {
  name: string;
}
interface dataItem { }

interface tree_module {
  query: Query;
  data: Array<dataItem>;
  currentItem?: dataItem;
  currentDevice?: types.deviceItem;
  total: number;
  defaultProps: { children: String; label: String };
  filterNode: (value: string, data: Tree) => boolean;
  pageChange: (val: number) => void;
  onSearch: () => void;
}
const tree_module = reactive<tree_module>({
  query: {
    name: '',
    pageIndex: 1,
    pageSize: 20,
    order: 'asc',
    orderBy: '',
  },
  data: [],
  total: 0,
  filterNode: (value: string, data: Tree) => {
    if (!value) return true;
    return data.label.includes(value);
  },
  // 分页导航
  pageChange: (val: number) => {
    tree_module.query.pageIndex = val;
    getData();
  },
  onSearch: () => {
    getData();
  },
  defaultProps: {
    children: 'devices',
    label: 'name',
  },
});
// 获取表格数据
const getData = () => {
  showLoading();
  api
    .list_svc(tree_module.query)
    .then((res) => {
      if (res.code == 0) {
        for (let i = 0; i < res.data.length; i++) {
          //如果 devices 只有1条，移动值 为 res.data[i] 属性
          if (res.data[i].devices && res.data[i].devices.length == 0) {
            delete res.data[i].devices;
          }
          else if (res.data[i].devices && res.data[i].devices.length == 1) {
            res.data[i].device = res.data[i].devices[0];
            delete res.data[i].devices;
          }
        }
        tree_module.data = res.data;
        tree_module.total = res.total || -1;
      } else {
        ElMessage.error(res.message);
      }
    })
    .finally(() => {
      closeLoading();
    });
};
const hasData = computed(() => tree_module.data.length > 0);
getData();

const onNodeCheck = (row?: any) => {
  tree_module.currentItem = row;
  //只有一个设备的点
  if (row.device) {
    let device = row.device;
    tree_module.currentDevice = device;
    let stream = device.streams;
    emits("nodeClick", stream, device)
  }
  //有多个设备的点 ，仅展开
  else if (row.devices) console.info('展开');
  else {
    //点位
    tree_module.currentDevice = row;
    emits("nodeClick", row.streams, row)
  }
};

</script>
<style lang="less" scoped>
::v-deep .el-card__body {
    padding-top: 5px;
    height: 100%;
    .content {
        padding: 0; 
        
        .el-tree-node__content {
            margin-left: -24px;
        }

        .label {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            padding: 0 5px;
        }
    }
}
</style>