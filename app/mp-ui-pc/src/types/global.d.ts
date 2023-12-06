declare interface IResponse<T = any> { 
    code: number
    message:string 
    data: T extends any ? T : T & any 
}
declare interface IPageResponse <T=any> extends IResponse<T>{
  total:number //number 与 Number 区别
}
declare interface IpageParam<T=any>{
  pageIndex:number
  pageSize:number
  orderBy?:String
  order?:String //[desc|asc]
  data?:T extends any?T:T&any
} 

interface sideBarItem{
  icon:string ,
  index:string,
  title: string,
  permiss: string,
  subs?:Array<sideBarItem>
}

interface ProcessQuery extends IpageParam{
  boatName:String,
  flowStatus:String,
  auditUser?:Number,
  manualAuditStatus:String,
  includeAuditStateNull:boolean,
  auditStateEq?:Boolean,
  breakRules:String
  datetimes:Array<string>, 
  pageIndex: Number,
  pageSize: Number,
  order:String,
  orderBy:String,
  groupIds:Array<Number>,
  boatSerials:Array<String>, 
  ipCameraSerial:Array<String> 
}

interface table_module{
  queryMoreOption:boolean,
  query:ProcessQuery,
  isResearch:boolean,
  cache:any,
  currentItem:any,
  tableData:any,
  pageTotal:number,
  treeData:Array<any>,
	treeCacheData:any,  
	treeDataMap:any
}
interface ProcessTableItem {
	id: number,
	webRecordTime:string,
  boatName:string,
  vioName:string,
  flowStatus:number,
  manualAuditResult:number, 
  videoSavePath:string,
  pic1SavePath:string,

  annoPic1SavePath:string,
  annoVideoSavePath:string,
  programAuditResult:number,
  manualAuditRemark:string,
  auditUser:number
}
//下列表 option 选项
interface optionItem{
  key:string,value?:number,label:string
}
interface KeyValue{
  key?:string,value?:string 
}
interface ItemAattachData2{
  flow_status:Array<optionItem> //[ ...optionItem] 
}
//Process 附加属性
interface ItemAattachData{
  allowAuditStatus:Array<number>,
  flow_status:Array<optionItem>, 
  getFlowStateName:(v:number)=>optionItem  ,
  manual_audit_state:Array<optionItem>, 
  getManualStateName:(v:number)=>optionItem,
  program_audit_state:Array<optionItem>, 
  getAutoStateName:(v:number)=>optionItem,
  statue2TagType:(v?:number)=>"" | "success" | "warning" | "info" | "danger",

  rule:Array<KeyValue>,  
  user_name_list:Array<optionItem>, 
}
 

interface download_config{
  method?:string,
  headers?:AxiosRequestHeaders
}
declare module 'vue3-video-play'
declare module 'wangeditor'
declare module  "json-bigint"





