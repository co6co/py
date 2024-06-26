import processDataView from './src/processDataView.vue'
import tree from './src/tree.vue'
import selectTree from './src/selectTree.vue'
import * as types from './src/types'
import type { IPageParam, Table_Module_Base } from 'co6co'

export { processDataView, tree, selectTree, types }

export interface Query extends IPageParam {
  id?: number
  boatName: string
  flowStatus: string
  auditUser?: number
  manualAuditStatus: string
  auditState: string
  includeAuditStateNull: boolean
  auditStateEq?: boolean
  breakRules: string
  datetimes: Array<string>
  groupIds: Array<number>
  boatSerials: Array<string>
  ipCameraSerial: Array<string>
}

export interface Item {
  index: number //什么值
  id: number
  webRecordTime: string
  boatName: string
  vioName: string
  flowStatus: number
  manualAuditResult: number
  videoSavePath: string
  pic1SavePath: string

  annoPic1SavePath: string
  annoVideoSavePath: string
  programAuditResult: number
  manualAuditRemark: string
  auditUser: number
}
export interface table_module extends Table_Module_Base {
  query: Query
  queryMoreOption: boolean
  isResearch: boolean
  cache: any
  currentItem?: Item
  tableData: Item[]
  treeData: types.TreeItem[]
  treeCacheData: any
  treeDataMap: any
}
