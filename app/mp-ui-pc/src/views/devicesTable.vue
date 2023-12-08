<template>
	<div> 
		<div class="container"> 
			<el-row :gutter="24">  
				<el-col :span="3"> 
					<el-input v-model="deviceName" placeholder="设备名称" /> 
					<el-tree @node-click="onNodeCheck"
						ref="tree"
						class="filter-tree"
						:data="tree_module.data"
						:props="tree_module.defaultProps"
						default-expand-all
						:filter-node-method="tree_module.filterNode"
					/> 
				</el-col>
				<el-col :span="15"> 
					<stream :sources="player.sources"></stream> 
				</el-col>
				<el-col :span="6"><ptz @ptz="OnPtz"></ptz></el-col> 
			</el-row> 
		</div> 
	</div>
</template>

<script setup lang="ts" name="basetable">
import { ref, watch,reactive, watchEffect, nextTick, PropType ,onMounted, onBeforeUnmount, computed  } from 'vue';
import { ElMessage, ElMessageBox,FormRules,FormInstance,ElTreeSelect } from 'element-plus';
import { TreeNode } from 'element-plus/es/components/tree-v2/src/types'
import { TreeNodeData } from 'element-plus/es/components/tree/src/tree.type'
import { Delete, Edit, Search, Compass,MoreFilled,Download } from '@element-plus/icons-vue'; 
import  * as api from '../api/device'; 
import  { stream ,ptz} from '../components/stream';   
 

const deviceName = ref("");
const tree=ref(null)

interface Tree {
  [key: string]: any
}
interface Query {
  name:string, 
}
interface dataItem{

}
interface tree_module  {
    query:Query,  
    data:Array<dataItem>,
    currentItem?:dataItem,
    total:number,
	defaultProps:{children:String,label:String}, 
	filterNode:(value: string,data:Tree)=>boolean
}  
const tree_module = reactive<tree_module>({ 
	query:{ 
		name:""  
	}, 
    data:[],
	total:-1,  
	filterNode : (value: string, data: Tree) => {
		if (!value) return true
		return data.label.includes(value)
	}, 
	defaultProps :{
		children: 'children',
		label: 'name',
	},
}); 

 
 
// 查询操作
const onSearch = () => { 
	getData();
};  
const getQuery=()=>{ 
}
// 获取表格数据
const getData = ( ) => {   
	getQuery(),api.list_svc(tree_module.query).then(res => {
        if (res.code==0){
            tree_module.data = res.data; 
		    tree_module.total = res.total || -1; 
        }else{
            ElMessage.error(res.message); 
        } 
	});
};
getData();   
/** 播放器 */
interface player_sources{  
	sources:Array< stream_source>
} 
const player=reactive<player_sources>({sources:[]})
 
const onNodeCheck=( item?:any)=>{ 
	tree_module.currentItem=item 
	player.sources= [
	
	{url:`http://127.0.0.1:18000/flv/vlive/${item.ip}.flv`,name:"HTTP-FLV"}, 
		{url:`ws://127.0.0.1:18000/ws-flv/vlive/${item.ip}.flv`,name:"WS-FLV"}, 
		{url:`webrtc://127.0.0.1:18000/rtc/vlive/${item.ip}`,name:"webrtc"}, 
		{url:`http://127.0.0.1:18000/vhls/${item.ip}/${item.ip}_live.m3u8`,name:"HLS(m3u8)"}
	]   
}
/** ptz */
const OnPtz=(name:string,type:string)=>{
	console.warn(name,type)
}
//**end 打标签 */
</script> 
<style  lang="less"> 
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
      color: #F56C6C;
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
  ::v-deep  .el-table tr,.el-table__row {cursor: pointer;  } 
  
  .formItem {
    display: flex;
    align-items: center;
    display: inline-block;
    .label{display: inline-block; color: #aaa; padding: 0 5px;}
  }
     
  ::v-deep .el-dialog__body{height:70%; overflow: auto;}
  .menuInfo{.el-menu{width: auto; .el-menu-item{padding: 10px; height: 40px;}}}
  
  /**
  
  ::v-deep .el-dialog{
      .el-dialog__header{padding: 5px;}
      .el-dialog__body{padding: 15px 5px;}
      .el-dialog__footer{padding: 5px;}
  } */
</style>
