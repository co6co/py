<template>
  <div class="header">
    <!-- 折叠按钮 -->
    <div class="collapse-btn" @click="collapseChage">
      <el-icon v-if="sidebar.collapse"><Expand /></el-icon>
      <el-icon v-else><Fold /></el-icon>
    </div>
    <div class="logo">
      {{ systeminfo.name }} <span class="version"> {{ systeminfo.version }}</span>
    </div>
    <div class="header-right">
      <div class="header-user-con">
        <!-- 消息中心 -->
        <div class="btn-bell" @click="router.push('/tabs')">
          <el-tooltip
            effect="dark"
            :content="message ? `有${message}条未读消息` : `消息中心`"
            placement="bottom"
          >
            <i class="el-icon-lx-notice"></i>
          </el-tooltip>
          <span class="btn-bell-badge" v-if="message"></span>
        </div>
        <!-- 用户头像 -->
        <el-avatar class="user-avator" :size="30" :src="userInfo.avatar.value" />
        <!-- 用户名下拉菜单 -->
        <el-dropdown class="user-name" trigger="click" @command="handleCommand">
          <span class="el-dropdown-link">
            {{ userName }}
            <el-icon class="el-icon--right">
              <arrow-down />
            </el-icon>
          </span>
          <template #dropdown>
            <el-dropdown-menu>
              <!--
              <a href="https://github.com/co6co" target="_blank">
                <el-dropdown-item>项目仓库</el-dropdown-item>
              </a>
              -->
              <el-dropdown-item command="user">个人中心</el-dropdown-item>
              <el-dropdown-item command="log">更新日志</el-dropdown-item>
              <el-dropdown-item divided command="loginout">退出登录</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>
      <history-dialog ref="dialogRef" style="width: 50%; height: 60%" />
    </div>
  </div>
</template>
<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useSidebarStore } from '../store/sidebar'
import { useRouter } from 'vue-router'
import defaultAvatar from '../assets/img/img.jpg'
import { Storage, removeAuthonInfo } from 'co6co'
import { useUserHook } from 'co6co-right'
import useSystem from '../hooks/useSystem'
import historyDialog from './log'
import { getUserName } from 'co6co'
const userName = ref(getUserName())
const message: number = 2
const storage = new Storage()
const sidebar = useSidebarStore()
const { systeminfo } = useSystem()
// 侧边栏折叠
const collapseChage = () => {
  sidebar.handleCollapse()
}
const userInfo = useUserHook.getUserInfo()
onMounted(() => {
  userInfo.avatar.value = userInfo.avatar.value || defaultAvatar
})
onMounted(() => {
  //屏幕宽度小于 1500 收起
  if (document.body.clientWidth < 1500) {
    collapseChage()
  }
})
const dialogRef = ref<InstanceType<typeof historyDialog>>()
// 用户名下拉菜单选择事件
const router = useRouter()
const handleCommand = (command: string) => {
  switch (command) {
    case 'loginout':
      removeAuthonInfo()
      localStorage.removeItem('ms_username')
      storage.remove('useRouteData')
      router.push('/login')
      break
    case 'user':
      router.push('/user')
      break
    case 'log':
      //console.info(dialogRef)
      dialogRef.value?.openDialog()
  }
}
</script>
<style scoped>
.header {
  position: relative;
  box-sizing: border-box;
  width: 100%;
  height: 70px;
  font-size: 22px;
  color: #fff;
}
.collapse-btn {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  float: left;
  padding: 0 21px;
  cursor: pointer;
}
.header .logo {
  float: left; 
  line-height: 70px;
  font-size: 90%;
  font-weight: bold; 
}
.header .version {
  font-size: 60%;
  color: #c0c0c0;
}
.header-right {
  float: right;
  padding-right: 50px;
}
.header-user-con {
  display: flex;
  height: 70px;
  align-items: center;
}
.btn-fullscreen {
  transform: rotate(45deg);
  margin-right: 5px;
  font-size: 24px;
}
.btn-bell,
.btn-fullscreen {
  position: relative;
  width: 30px;
  height: 30px;
  text-align: center;
  border-radius: 15px;
  cursor: pointer;
  display: flex;
  align-items: center;
}
.btn-bell-badge {
  position: absolute;
  right: 4px;
  top: 0px;
  width: 8px;
  height: 8px;
  border-radius: 4px;
  background: #f56c6c;
  color: #fff;
}
.btn-bell .el-icon-lx-notice {
  color: #fff;
}
.user-name {
  margin-left: 10px;
}
.user-avator {
  margin-left: 20px;
}
.el-dropdown-link {
  color: #fff;
  cursor: pointer;
  display: flex;
  align-items: center;
}
.el-dropdown-menu__item {
  text-align: center;
}
</style>
