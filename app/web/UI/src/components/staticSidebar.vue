<template>
  <div class="sidebar">
    <el-menu
      class="sidebar-el-menu"
      :default-active="onRoutes"
      :collapse="sidebar.collapse"
      background-color="#324157"
      text-color="#bfcbd9"
      active-text-color="#20a0ff"
      unique-opened
      router
    >
      <template v-for="item in items">
        <template v-if="item.subs">
          <el-sub-menu :index="item.index" :key="item.index" v-permiss="item.permiss">
            <template #title>
              <el-icon>
                <component :is="item.icon"></component>
              </el-icon>
              <span>{{ item.title }}</span>
            </template>
            <template v-for="(subItem, index) in item.subs">
              <el-sub-menu
                v-if="subItem.subs"
                :index="subItem.index"
                :key="subItem.index"
                v-permiss="item.permiss"
              >
                <template #title>
                  {{ subItem.title }}
                </template>
                <el-menu-item
                  v-for="(threeItem, i) in subItem.subs"
                  :key="i"
                  :index="threeItem.index"
                >
                  {{ threeItem.title }}
                </el-menu-item>
              </el-sub-menu>
              <el-menu-item v-else :index="subItem.index" :key="index" v-permiss="subItem.permiss">
                <template #title>
                  <el-icon>
                    <component :is="item.icon"></component>
                  </el-icon>
                  {{ subItem.title }}
                </template>
              </el-menu-item>
            </template>
          </el-sub-menu>
        </template>
        <!--没有子菜单-->
        <template v-else>
          <el-menu-item :index="item.index" :key="item.index" v-permiss="item.permiss">
            <el-icon>
              <component :is="item.icon"></component>
            </el-icon>
            <template #title>{{ item.title }}</template>
          </el-menu-item>
        </template>
      </template>
    </el-menu>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useSidebarStore } from '../store/sidebar'
import { useRoute } from 'vue-router'
import { routeHook } from 'co6co-right'

const items: Array<routeHook.sideBarItem> = [
  {
    icon: 'Warning',
    index: '/usermgr',
    title: '用户管理',
    permiss: 'admin'
  },
  {
    icon: 'Calendar',
    index: '/processRecord',
    title: '记录查询',
    permiss: 'admin'
  },
  {
    icon: 'HelpFilled',
    index: '/processAudit',
    title: '审核系统',
    permiss: 'user'
  },
  {
    icon: 'DocumentCopy',
    index: '/taskTable',
    title: '任务列表',
    permiss: 'admin'
  },
  {
    icon: 'Edit',
    index: '/groupTable',
    title: '分组信息',
    permiss: 'admin'
  },

  {
    icon: 'Calendar',
    index: '1',
    title: '标签管理',
    permiss: 'admin',
    subs: [
      { icon: 'DocumentCopy', index: '/labelTable', title: '标签管理', permiss: 'admin' },
      { icon: 'PieChart', index: '/processLable', title: '数据标签', permiss: 'admin' }
    ]
  },

  {
    icon: 'Menu',
    index: '2',
    title: '系统管理',
    permiss: 'user',
    subs: [
      {
        icon: 'Link',
        index: '/userBoat',
        title: '关联船',
        permiss: 'admin'
      },
      {
        icon: 'Aim',
        index: '/userJobs',
        title: '审核任务',
        permiss: 'user'
      },
      {
        icon: 'Grid',
        index: '/ruleTable',
        title: '规则管理',
        permiss: 'admin'
      },
      {
        icon: 'Tools',
        index: '/auditSetting',
        title: '审核配置',
        permiss: 'admin'
      },
      {
        icon: 'Tools',
        index: '/groupTree',
        title: '船优先级',
        permiss: 'admin'
      }
    ]
  },
  {
    icon: 'Menu',
    index: '3',
    title: '系统维护',
    permiss: 'admin',
    subs: [
      {
        icon: 'Tools',
        index: '/userGroup',
        title: '用户组',
        permiss: 'admin'
      },
      {
        icon: 'Tools',
        index: '/menu',
        title: '系统菜单',
        permiss: 'admin'
      },
      {
        icon: 'Tools',
        index: '/role',
        title: '角色管理',
        permiss: 'admin'
      }
    ]
  }
  /**
    {
        icon: 'Odometer',
        index: '/dashboard',
        title: '系统首页',
        permiss: '1',
    },
    {
        icon: 'Calendar',
        index: '1',
        title: '表格相关',
        permiss: '2',
        subs: [
            {
                index: '/table',
                title: '常用表格',
                permiss: '2',
            },
            {
                index: '/import',
                title: '导入Excel',
                permiss: '2',
            },
            {
                index: '/export',
                title: '导出Excel',
                permiss: '2',
            },
            {
                index: '/usermgr',
                title: '用户管理',
                permiss: '2',
            },
            {
                index: '/processAudit',
                title: '数据审核',
                permiss: '2',
            },
        ],
    },
    {
        icon: 'DocumentCopy',
        index: '/tabs',
        title: 'tab选项卡',
        permiss: '3',
    },
    {
        icon: 'Edit',
        index: '3',
        title: '表单相关',
        permiss: '4',
        subs: [
            {
                index: '/form',
                title: '基本表单',
                permiss: '5',
            },
            {
                index: '/upload',
                title: '文件上传',
                permiss: '6',
            },
            {
                index: '4',
                title: '三级菜单',
                permiss: '7',
                subs: [
                    {
                        index: '/editor',
                        title: '富文本编辑器',
                        permiss: '8',
                    },
                    {
                        index: '/markdown',
                        title: 'markdown编辑器',
                        permiss: '9',
                    },
                ],
            },
        ],
    },
    {
        icon: 'Setting',
        index: '/icon',
        title: '自定义图标',
        permiss: '10',
    },
    {
        icon: 'PieChart',
        index: '/charts',
        title: 'schart图表',
        permiss: '11',
    },
    {
        icon: 'Warning',
        index: '/permission',
        title: '权限管理',
        permiss: '13',
    },
    {
        icon: 'CoffeeCup',
        index: '/donate',
        title: '支持作者',
        permiss: '14',
    },
     */
]

const route = useRoute()
const onRoutes = computed(() => {
  return route.path
})

const sidebar = useSidebarStore()
</script>

<style scoped>
.sidebar {
  display: block;
  position: absolute;
  left: 0;
  top: 70px;
  bottom: 0;
  overflow-y: scroll;
}
.sidebar::-webkit-scrollbar {
  width: 0;
}
.sidebar-el-menu:not(.el-menu--collapse) {
  width: 250px;
}
.sidebar > ul {
  height: 100%;
}
</style>
