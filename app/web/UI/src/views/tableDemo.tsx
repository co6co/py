import type { TableInstance } from 'element-plus'
import { ElTable, ElTableColumn, ElButton } from 'element-plus'

import { ref, defineComponent, VNodeChild } from 'vue'
import { TableView } from 'co6co-right'
import { get_table_svc } from '@/api/biz/task'
export default defineComponent({
  setup(prop, ctx) {
    interface User {
      id: number
      date: string
      name: string
      address: string
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
    //:page reader
    const rander = (): VNodeChild => {
      return (
        <div>
          <ElTable
            ref="multipleTableRef"
            data={tableData}
            style="width: 100%"
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
