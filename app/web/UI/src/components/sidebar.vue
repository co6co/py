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
      <template v-for="item in items.data">
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
                    <component :is="subItem.icon"></component>
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
import { computed, reactive, nextTick, onMounted } from 'vue'
import { useSidebarStore } from '../store/sidebar'
import { useRoute } from 'vue-router'
import { routeHook, menuSelectHook } from 'co6co-right'

const { getRouteData } = routeHook.useRouteData()
const default_icon = 'Calendar'
const convertSideItem = (
  data: routeHook.IRouteData,
  subs?: routeHook.sideBarItem[]
): routeHook.sideBarItem => {
  const result = {
    icon: data.icon ? data.icon : default_icon,
    index: data.url ? data.url : data.code,
    title: data.name,
    //permiss:"" //页面似乎不需要权限字 有就能访问
    permiss: data.permissionKey
  }
  if (!subs || subs.length == 0) return result
  return { ...result, ...{ subs: subs } } as routeHook.sideBarItem
}
const routeData2Item = (data: routeHook.IRouteData): routeHook.sideBarItem | null => {
  if (
    (data.category == menuSelectHook.MenuCateCategory.GROUP ||
      data.category == menuSelectHook.MenuCateCategory.VIEW) &&
    data.children &&
    data.children.length > 0
  ) {
    let sub = data.children.map((d) => routeData2Item(d)).filter((m) => m)
    if (data.category == menuSelectHook.MenuCateCategory.GROUP && (!sub || sub.length == 0))
      return null
    return convertSideItem(data, sub as routeHook.sideBarItem[])
  }
  if (
    data.category == menuSelectHook.MenuCateCategory.VIEW &&
    (!data.children || data.children.length == 0)
  ) {
    return convertSideItem(data)
  }
  return null
}

const items = reactive<{ data: routeHook.sideBarItem[] }>({ data: [] })
const getData = () => {
  nextTick(() => {
    const routeData = getRouteData()
    let data: routeHook.sideBarItem[] = []
    if (routeData && routeData.length > 0) {
      routeData.forEach((i) => {
        const item = routeData2Item(i)
        if (item) data.push(item)
      })
      items.data = data
    }
  })
}
onMounted(() => {
  getData()
})
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
