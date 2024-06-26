 
//下列表 option 选项
interface optionItem{
  key:string,value?:number,label:string
}
interface KeyValue{
  key?:string,value?:string 
}
interface ItemAattachData2{
  flow_status:[ ...optionItem] 
}
//Process 附加属性
interface ItemAattachData{
  allowAuditStatus:[...number],
  flow_status:[ ...optionItem], 
  getFlowStateName:(v:number)=>optionItem  ,
  manual_audit_state:[ ...optionItem],
  ai_audit_state:[ ...optionItem],
  getManualStateName:(v:number)=>optionItem,
  program_audit_state:[...optionItem ],
  getAutoStateName:(v:number)=>optionItem,
  statue2TagType:(v?:number)=>"" | "success" | "warning" | "info" | "danger",

  rule:[...KeyValue],
  user_name_list:[ ...optionItem], 
}
 

interface download_config{
  method?:string,
  headers?:AxiosRequestHeaders
}
declare module 'vue3-video-play'
declare module 'wangeditor'
declare module  "json-bigint"
declare module "md-editor-v3"
declare module "vue-schart"