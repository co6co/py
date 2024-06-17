import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import * as api from '../api/pd'
import useUserState from './useUserState'
export enum NotifyType {
  /**
   * 自动推送 相关船发生变化
   * */
  auto_audit_boat_changed,
  /**
   * 规则 人工复审状态改变
   * */
  manual_review_state_changed,
  /**
   * 规则 优先级 改变
   * */
  rule_priority_changed,
  /**
   * 船 优先级 改变
   */
  boat_priority_changed,
  /**
   * 用户分配到的船 改变
   */
  related_boat_changed
}
export interface INotifyOption {
  type: NotifyType,
  state: any,
  failMessage?: string,
  userId?: number,
  message?: string,
}
export default function () {
  const privitiveState = ref(false)
  const setOldState = (value: any) => {
    privitiveState.value = value
  }
  const { userModel } = useUserState()

  /**
   * 通知审核系统
   *
   *
   *
   * @param type 通知类型
   * @param userId 通知者
   * @param state 新的状态值 oldState != state 通知 审核系统
   * @param 需要改变的用户，默认为 null 取当前用户
   */
  const notifyAuditSystem =  (
    option:INotifyOption={type:NotifyType.auto_audit_boat_changed, state:true} 
  ) => {
    let svc
    //值未改变 ，不操作
    if (privitiveState.value ==option. state) return
    switch (option.type) {
      case NotifyType.auto_audit_boat_changed:
        svc = api.auto_audit_boat_changed_svc
        break
      case NotifyType.manual_review_state_changed:
        svc = api.manual_review_state_changed_svc
        break
      case NotifyType.rule_priority_changed:
        svc = api.rule_priority_changed_svc
        break
      case NotifyType.boat_priority_changed:
        svc = api.boat_priority_changed_svc
        break
      case NotifyType.related_boat_changed:
        svc = api.related_boat_changed_svc
        break
    }
    if (svc) {
      if (!option.userId) option.userId = userModel.id
      svc(option.userId).then((res)=>{
        if (api.isSuccess(res)) ElMessage.success(res.message ||option. message || '通知成功');
        else { 
          ElMessage.error(res.message ||option. failMessage || '通知不成功')
        }
      })
      .catch((e)=>{
        ElMessage.error(option. failMessage || '通知不成功')
      })
     
    }
  }

  return { setOldState, notifyAuditSystem }
}
