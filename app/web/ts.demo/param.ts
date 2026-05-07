// 前置的不定量参数
type JSTypeMap={
    boolean:boolean,
    number:number,
    string:string,
    function :(...args:any[])=>any
}
type JStype=keyof JSTypeMap;
type getType<T extends JStype[]>={
    [I in keyof T]: JSTypeMap[T[I]]
}
declare function exec<T extends JStype[]>(...T:[...T,(...args:getType<T>)=>any]):void
exec('boolean','number','string',(a,b,c)=>{
    
});