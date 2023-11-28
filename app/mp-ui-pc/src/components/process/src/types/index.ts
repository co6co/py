export interface PropsIter {
    value:string
    label:string
    children:string
    disabled?:boolean
    isLeaf:string
}

//树形select  item
export interface TreeItem {
  id: string
  label: string
  children?: TreeItem[]
}