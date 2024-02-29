<template >
    <img-video :viewOption="form2.data"  ></img-video>
</template>
<script setup lang="ts">
import { onMounted, ref } from 'vue';
import  { imgVideo,types} from '../components/player';    
import  * as res_api from '../api';   
import { type AlarmItem,getResources } from '../components/biz';
 
import {useAppDataStore} from '../store/appStore'
const dataStore=useAppDataStore() 

onMounted(()=>{
    const rowData:AlarmItem=dataStore.getState( )  
    loadData(rowData)

})
//**媒体阅览 */
interface dialog2DataType{ 
	data:Array<types.resourceOption>
}
let dialog2Data={ 
	data:[]
}
let form2 = ref<dialog2DataType>(dialog2Data);  
const getResultUrl=(uuid:string,isposter:boolean=false)=>{
	if (isposter) return import.meta.env.VITE_BASE_URL+`/api/resource/poster/${uuid}/700/600`;
	return import.meta.env.VITE_BASE_URL+`/api/resource/${uuid}`
}
const loadData=(row:AlarmItem)=>{  
	form2.value.data=getResources(row)
} 
</script>