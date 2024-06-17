import { get_status_svc } from '../../api/process'
import user_api from '../../api/sys/user'
import { type ISelect } from 'co6co'

const res = await get_status_svc()
const user_res = await user_api.get_select_svc()
const data: {
  allowAuditStatue: Array<number>
  flowStatues: Array<optionItem>
  rule: Array<KeyValue>
} = res.data
const labels: Array<any> = []
labels.push({ key: '', value: '', label: '--处理状态--' })
labels.push(...data.flowStatues)

const rule = []
rule.push({ key: '', value: '---违反规则---' })
rule.push(...data.rule)

const namelist: Array<ISelect> = user_res.data
const user_name_list: Array<any> = []
user_name_list.push({ key: '', value: '', label: '--审核人员--' })
user_name_list.push(
  ...namelist.map(function (m: any) {
    return { key: m.userName, value: m.id, label: m.name }
  })
)

export const form_attach_data: ItemAattachData = {
  allowAuditStatus: data.allowAuditStatue,
  flow_status: labels,
  manual_audit_state: [
    { key: '', value: '', label: '--人工审核状态--' },
    { key: '', value: 0, label: '未通过(误警)' },
    { key: '', value: 1, label: '通过(确警)' },
    { key: '', value: 2, label: '通过并下发给用户(确警)' },
    { key: '', value: 3, label: '不做处理' }
  ],
  ai_audit_state: [
    { key: '', value: '', label: '--AI审核状态--' },
    { key: '', value: 0, label: '未通过(误警)' },
    { key: '', value: 1, label: '通过(确警)' },
    { key: '', value: 2, label: '通过并下发给用户(确警)' },
    { key: '', value: 3, label: '不做处理' }
  ],
  program_audit_state: [
    { key: '', value: 0, label: '未通过(误警)' },
    { key: '', value: 1, label: '通过(确警)' }
  ],
  user_name_list: user_name_list,
  rule: rule,
  getFlowStateName(v: number) {
    return this.flow_status.find((m) => m.value === v)
  },
  getManualStateName(v: number) {
    if (v == null) return { key: '', value: '', label: '待审核' }
    return this.manual_audit_state.find((m) => m.value === v)
  },
  getAutoStateName(v: number) {
    if (v == null) return { key: '', value: '', label: '待审核' }
    return this.program_audit_state.find((m) => m.value === v)
  },
  statue2TagType(v?: number) {
    switch (v) {
      case null:
        return 'info'
      case 0:
        return 'danger'
      //case 1:return 'primary'
      case 2:
        return 'success'
      case 3:
        return 'warning'
      default:
        return '' //primary
    }
  }
}
