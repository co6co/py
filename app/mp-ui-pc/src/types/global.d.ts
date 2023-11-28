
declare interface IResponse<T = any> { 
    code: int
    message:string 
    data: T extends any ? T : T & any 
}
declare interface IPageResponse <T=any> extends IResponse<T>{
  total:int
}
declare interface IpageParam<T=any>{
  pageIndex:int
  pageSize:int
  orderBy?:string
  order?:string //[desc|asc]
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
  treeData:ref<types.TreeItem[]>,
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
  flow_status:[ ...optionItem] 
}
//Process 附加属性
interface ItemAattachData{
  allowAuditStatus:[...number],
  flow_status:[ ...optionItem], 
  getFlowStateName:(v:number)=>optionItem  ,
  manual_audit_state:[ ...optionItem],
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