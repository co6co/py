import { get_config_svc } from '../api/wx';
const res =await get_config_svc()
const data:{group:Array<optionItem>,postion:Array<optionItem>,allowSetNumberGroup:Array<string>}=res.data;

let group:Array<any>=[];
group.push({key:"",value:'',label:"--分组类型--"})
group.push(...data.group) 

let postion:Array<any>=[];
postion.push({key:"",value:'',label:"--安装位置--"})
postion.push(...data.postion)  

export interface AtachData{ 
    group:Array<optionItem>, 
    getGroupItem:(v:string)=>optionItem|undefined,
    postion:Array<optionItem>,
    getPostionItem:(v:number)=>optionItem|undefined,  
    allowSetNumberGroup:Array<string>,
    statue2TagType:(v?:number)=>"" | "success" | "warning" | "info" | "danger",
}
   
export const attach_data:AtachData={  
	group:group,  
	postion:postion,
	allowSetNumberGroup:data.allowSetNumberGroup,
	getGroupItem(v:string){ 
        if (v==null)return {key:"",value:undefined,label:""}
        return this.group.find(m=>m.key=== v )
    },
	getPostionItem(v:number){  
		if (v==null)return {key:"",value:undefined,label:""} 
        return this.postion.find(m=>m.value===v ) 
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