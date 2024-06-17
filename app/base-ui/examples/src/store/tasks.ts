import { get_status_svc } from '../api/tasks';
const res =await get_status_svc()
const data:{statue:Array<optionItem>,types:Array<optionItem>,downloadTask:number}=res.data;
const statue:Array<any>=[];
statue.push({key:"",value:'',label:"--任务状态--"})
statue.push(...data.statue) 

const types:Array<any>=[];
types.push({key:"",value:'',label:"--任务类型--"})
types.push(...data.types)  
export interface TaskAtachData{ 
    tast_status:Array<optionItem>, 
    getStateItem:(v:number)=>optionItem|undefined,
    task_types:Array<optionItem>,
    getTypeItem:(v:number)=>optionItem|undefined, 
    statue2TagType:(v?:number)=>"" | "success" | "warning" | "info" | "danger",
    downTask:number,
}

export const attach_data:TaskAtachData={  
	tast_status:statue,  
	task_types:types,
	downTask:data.downloadTask,
	getStateItem(v:number){ 
        if (v==null)return {key:"",value:undefined,label:""}
        return this.tast_status.find(m=>m.value=== v )
    },
	getTypeItem(v:number){  
		if (v==null)return {key:"",value:undefined,label:""} 
        return this.task_types.find(m=>m.value===v ) 
    },
	 
	statue2TagType(v?:number){ 
		switch(v){
			case null:return 'info'
			case 0:return 'danger'
			//case 1:return 'primary'
			case 2:return 'success'
			case 3:return 'warning' 
			default:return "" //primary
		}
	}
 }