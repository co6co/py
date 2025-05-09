import type { TableInstance } from 'element-plus'
import { ElTable, ElTableColumn, ElButton, ElMessageBox, ElMessage } from 'element-plus'

import { ref, defineComponent, VNodeChild } from 'vue'
import { TableView } from 'co6co-right'
import { get_table_svc } from '@/api/dev'
import { useRouter } from 'vue-router'
export default defineComponent({
  setup(prop, ctx) {
    interface User {
      id: number
      date: string
      name: string
      address: string
    }

    function blobToText(blob: Blob) {
      /*
			return new Promise((resolve, reject) => {
				const reader = new FileReader();
				reader.onload = () => resolve(reader.result);
				reader.onerror = reject;
				reader.readAsText(blob); // 默认编码是 UTF-8
			});
			*/
      const reader = new FileReader()
      reader.onload = (event) => {
        ElMessage.success(event.target!.result as string)
      }
      reader.readAsText(blob) // 默认编码是 UTF-8
    }

    const router = useRouter()
    const onGotoFileView = () => {
      const param = {
        name: 'sys_file_view',
        state: {
          params: { path: 'I:\\56BDF36964CD' }
        }
      }
      router.push(param)
    }
    const multipleTableRef = ref<TableInstance>()
    const multipleSelection = ref<User[]>([])

    const selectable = (row: User) => ![1, 2].includes(row.id)

    const handleSelectionChange = (val: User[]) => {
      console.info('changed', val)
      multipleSelection.value = val
    }

    const tableData: User[] = [
      {
        id: 1,
        date: '2016-05-03',
        name: 'Tom',
        address: 'No. 189, Grove St, Los Angeles'
      },
      {
        id: 2,
        date: '2016-05-02',
        name: 'Tom',
        address: 'No. 189, Grove St, Los Angeles'
      },
      {
        id: 3,
        date: '2016-05-04',
        name: 'Tom',
        address: 'No. 189, Grove St, Los Angeles'
      },
      {
        id: 4,
        date: '2016-05-01',
        name: 'Tom',
        address: 'No. 189, Grove St, Los Angeles'
      },
      {
        id: 5,
        date: '2016-05-08',
        name: 'Tom',
        address: 'No. 189, Grove St, Los Angeles'
      },
      {
        id: 6,
        date: '2016-05-06',
        name: 'Tom',
        address: 'No. 189, Grove St, Los Angeles'
      },
      {
        id: 7,
        date: '2016-05-07',
        name: 'Tom',
        address: 'No. 189, Grove St, Los Angeles'
      }
    ]
    const onClick = () => {
      ElMessageBox.prompt('请输入', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        inputValue: '123'
      })
        .then(({ value }) => {
          ElMessage.success(`你输入的内容：${value}`)
        })
        .catch((re) => {
          ElMessage.info('取消' + re)
        })
    }
    //:page reader
    const rander = (): VNodeChild => {
      return (
        <div>
          <ElButton onClick={onClick}>测试</ElButton>
          <ElButton onClick={onGotoFileView}>跳转</ElButton>
          <ElTable
            ref="multipleTableRef"
            data={tableData}
            onSelection-change={handleSelectionChange}
          >
            <ElTableColumn type="selection" selectable={selectable} width={55} />
            <ElTableColumn property="name" label="Name" width="120" />
            <ElTableColumn property="address" label="Address" />
          </ElTable>

          <TableView
            dataApi={get_table_svc}
            style="width: 100%"
            onSelection-change={handleSelectionChange}
          >
            {{
              default: () => (
                <>
                  <ElTableColumn type="selection" selectable={selectable} width={55} />
                  <ElTableColumn
                    label="编号"
                    prop="code"
                    align="center"
                    width={180}
                    sortable="custom"
                    showOverflowTooltip={true}
                  />
                  <ElTableColumn
                    label="名称"
                    prop="name"
                    align="center"
                    sortable="custom"
                    showOverflowTooltip={true}
                  />
                </>
              )
            }}
          </TableView>
        </div>
      )
    }
    return rander
  } //end setup
})
