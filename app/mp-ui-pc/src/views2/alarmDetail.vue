<template>
    <details-info :data="form.data"></details-info>
</template>
<script setup lang="ts">
import {
    ref,
    watch,
    reactive,
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
    ElTreeSelect,
    dayjs
} from "element-plus";
import { str2Obj } from "../utils";
import { detailsInfo } from "../components/details";
import { useRouter } from "vue-router";
import {useAppDataStore} from '../store/appStore'
const dataStore=useAppDataStore() 

onMounted(()=>{
    const router = useRouter();
    //const mode = router.currentRoute.value.query.mode;
    //const rowData = router.currentRoute.value.params
    const rowData=dataStore.getState( ) 
    console.info(rowData,"<===")
    loadData(rowData)
})
//**详细信息 */
interface dialogDataType {
    data: Array<any>;
}
let dialogData = {
    dialogVisible: false,
    data: []
};
let form = reactive<dialogDataType>(dialogData);
const loadData = (row?: any) => {
    if (row) {
        console.info(row)
        form.data = [
            { name: "检测结果信息", data: str2Obj(row.alarmAttachPO.result) },
            { name: "视频流信息", data: str2Obj(row.alarmAttachPO.media) },
            { name: "GPS信息", data: str2Obj(row.alarmAttachPO.gps) }
        ];
    }
}; 
</script>
