<template >
    <img-video :viewOption="form2.data"  ></img-video>
</template>
<script setup lang="ts">
import { onMounted, ref } from 'vue';
import  { imgVideo,types} from '../components/player';    
import  * as res_api from '../api';   
 
import {useAppDataStore} from '../store/appStore'
const dataStore=useAppDataStore() 

onMounted(()=>{ 
    //const mode = router.currentRoute.value.query.mode;
    //const rowData = router.currentRoute.value.params
    const rowData=dataStore.getState( )  
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
const setVideoResource=(uuid:string,option:types.videoOption)=>{
	res_api.request_resource_svc(import.meta.env.VITE_BASE_URL+`/api/resource/poster/${uuid}`).then(res=>{option.poster=res}).catch(e=>option.poster="" );  
	res_api.request_resource_svc(import.meta.env.VITE_BASE_URL+`/api/resource/${uuid}`).then(res=>{ option.url=res}).catch(e=>option.url="" );   
}
const setImageResource=(uuid:string,option:types.imageOption)=>{
	res_api.request_resource_svc(import.meta.env.VITE_BASE_URL+`/api/resource/${uuid}`).then(res=>{ option.url=res}).catch(e=>option.url="" );  
}
const getResultUrl=(uuid:string,isposter:boolean=false)=>{
	if (isposter) return import.meta.env.VITE_BASE_URL+`/api/resource/poster/${uuid}/700/600`;
	return import.meta.env.VITE_BASE_URL+`/api/resource/${uuid}`
}
const loadData=( row:any)=>{ 
	form2.value.data=[  
		{ url: getResultUrl(row.rawImageUid), name:"原始图片" ,poster:getResultUrl(row.rawImageUid,true),type:1},
		{ url:getResultUrl(row.markedImageUid),  name:"标注图片",poster:getResultUrl(row.markedImageUid,true), type:1} ,
	 	{ url:getResultUrl(row.videoUid),  name:"原始视频", poster:getResultUrl(row.videoUid,true),type:0},
	] 
} 
</script>