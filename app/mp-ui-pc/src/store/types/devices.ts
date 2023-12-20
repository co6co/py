export interface streamItem {name:string;url:string}
export interface dataItem {
    id: number;
    name: string;
    ip:string;
    innerIp:string; 
    streams?:Array<streamItem> 
}