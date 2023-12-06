import JSONbig  from 'json-bigint' 

export const str2Obj=(str:string)=>{
    return JSONbig.parse( str)
}

