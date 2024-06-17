import { ref, onMounted } from 'vue'
import * as api_type from 'co6co'
import { ElMessageBox, ElMessage } from 'element-plus'
import { showLoading, closeLoading } from '../../components/Logining'
export default function (del_svc: (id: number) => Promise<api_type.IResponse>, bck?: () => void) {
  const onDelete = (pk: any, name: string) => {
    // 二次确认删除
    ElMessageBox.confirm(`确定要删除"${name}"吗？`, '提示', {
      type: 'warning'
    })
      .then(() => {
        showLoading()
        del_svc(pk)
          .then((res) => {
            if (res.code == 0) {
              ElMessage.success('删除成功')
              if (bck) bck()
            } else ElMessage.error(`删除失败:${res.message}`)
          })
          .finally(() => {
            closeLoading()
          })
      })
      .catch(() => {})
  }
  return { onDelete }
}
