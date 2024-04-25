
type Collection<T> = (a: Array<T>, b: Array<T>) =>Array<T>
// 交集
export const intersect =(array1:[],array2:[])=>  array1.filter(x => array2.indexOf(x)>-1); 
 
// 差集 
export const minus:Collection<number>  =(array1:Array<number>,array2:Array<number>)=>  array1.filter(x => array2.indexOf(x)==-1); 
// 补集
export const complement =(array1:[],array2:[])=> {
	array1.filter(function(v){ return !(array2.indexOf(v) > -1) })
	.concat(array2.filter(function(v){ return !(array1.indexOf(v) > -1)}))
}
// 并集
let unionSet =(array1:[],array2:[])=> { 
	return array1.concat(array2.filter(function(v){ return !(array1.indexOf(v) > -1)}));
}