/*
Navicat MySQL Data Transfer

Source Server         : localhost
Source Server Version : 50721
Source Host           : localhost:3306
Source Database       : audit

Target Server Type    : MYSQL
Target Server Version : 50721
File Encoding         : 65001

Date: 2024-07-03 11:24:11
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for sys_menu
-- ----------------------------
DROP TABLE IF EXISTS `sys_menu`;
CREATE TABLE `sys_menu` (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `parent_id` int(11) DEFAULT NULL,
  `category` int(11) DEFAULT NULL COMMENT '0:后台URL,1:view,2:button',
  `icon` varchar(128) DEFAULT NULL,
  `name` varchar(64) DEFAULT NULL COMMENT '名称',
  `code` varchar(64) DEFAULT NULL COMMENT 'code',
  `url` varchar(255) DEFAULT NULL,
  `component` varchar(128) DEFAULT NULL,
  `methods` varchar(64) DEFAULT NULL COMMENT '方法名:GET|POST|DELETE|PUT|PATCH',
  `permission_key` varchar(64) DEFAULT NULL COMMENT 'view 与button 有效',
  `order` int(11) DEFAULT NULL,
  `status` int(11) DEFAULT NULL,
  `remark` varchar(255) DEFAULT NULL COMMENT '备注',
  `create_user` bigint(20) DEFAULT NULL COMMENT '创建人',
  `update_user` bigint(20) DEFAULT NULL COMMENT '修改人',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT NULL COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `code` (`code`)
) ENGINE=InnoDB AUTO_INCREMENT=79 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of sys_menu
-- ----------------------------
-- 根节点需要 手动改为0 插入会变成1
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('0', null, '0', null, '根节点', 'root', null, null, '', null, '0', '0', '12', '1', '1', '2024-04-28 14:44:24', '2024-04-28 14:44:37');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('3', '0', '2', 'Avatar', '用户管理', 'user', '/usermgr', '/views/permission/UserTableView.vue', null, 'admin', '1', '0', null, '1', '1', '2024-04-28 16:08:40', '2024-07-01 13:45:18');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('4', '3', '10', null, '用户增加', 'user_add', null, null, 'add', 'add', '1', '0', null, '1', '1', '2024-04-28 17:06:05', '2024-05-16 09:37:02');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('5', '3', '10', null, '用户编辑', 'user_edit', null, null, 'edit', 'edit', '2', '0', null, '1', '1', '2024-04-28 17:06:41', '2024-05-16 10:22:36');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('6', '3', '10', null, '用户删除', 'user_del', null, null, 'del', 'del', '3', '0', null, '1', '1', '2024-04-28 17:07:22', '2024-05-16 09:42:43');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('7', '3', '10', null, '重置密码', 'user_reset_passwd', null, null, 'reset', 'reset_pwd', '4', '0', null, '1', '1', '2024-04-28 17:08:10', '2024-05-16 09:42:51');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('8', '3', '3', 'Avatar', '修改当前密码', 'current_user_info', '/user', '../views/user.vue', '', 'user', '10', '0', '', '1', null, '2024-05-21 16:42:29', '2024-05-21 16:44:25');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('9', '3', '10', null, '用户关联角色', 'user_associated', null, null, 'associated', 'associated', '5', '0', null, '1', null, '2024-05-16 13:52:17', null);
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('20', '0', '0', 'Key', '系统维护', 'SYSTEM', null, null, null, 'admin', '8', '0', null, '1', '1', '2024-04-28 17:02:04', '2024-05-17 15:13:49');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('21', '20', '2', 'Menu', '系统菜单', 'menu', '/menu', '/views/permission/MenuTreeView.vue', null, 'admin', '1', '0', null, '1', '1', '2024-04-28 17:03:10', '2024-07-01 11:59:41');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('22', '20', '2', 'ColdDrink', '角色管理', 'role', '/role', '/views/permission/RoleView.vue', null, 'admin', '3', '0', null, '1', '1', '2024-04-29 16:34:19', '2024-07-01 12:00:13');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('23', '20', '2', 'Avatar', '用户组', 'userGroup', '/userGroup', '/views/permission/UserGroupTreeView.vue', null, 'supper', '2', '0', null, '1', '1', '2024-04-28 17:04:19', '2024-07-01 12:00:05');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('25', '0', '0', null, 'API管理', 'API_MGR', null, null, null, null, '999', '0', null, '1', '1', '2024-04-28 17:09:27', '2024-04-28 17:13:40');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('30', '21', '10', null, '增加系统菜单', 'menu_add', null, null, 'add', 'add', '1', '0', null, '1', null, '2024-05-16 11:22:37', null);
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('31', '21', '10', null, '编辑系统菜单', 'menu_edit', null, null, 'edit', 'edit', '2', '0', null, '1', null, '2024-05-16 11:23:24', null);
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('32', '21', '10', null, '删除菜单', 'del_menu', null, null, 'del', 'menu_del', '3', '0', null, '1', null, '2024-05-16 11:24:15', null);
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('33', '22', '10', null, '增加角色', 'role_add', null, null, 'add', 'add', '1', '0', null, '1', '1', '2024-05-16 11:25:22', '2024-05-16 11:27:39');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('34', '22', '10', null, '编辑角色', 'role_edit', null, null, 'edit', 'edit', '2', '0', null, '1', '1', '2024-05-16 11:26:10', '2024-05-16 11:35:15');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('35', '22', '10', null, '角色关联菜单', 'role_ass', null, null, 'associated', 'associated', '3', '0', null, '1', '1', '2024-05-16 11:27:17', '2024-05-16 11:35:32');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('36', '22', '10', null, '删除角色', 'role_del', null, null, 'del', 'del', '4', '0', null, '1', null, '2024-05-16 11:33:43', null);
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('37', '23', '10', null, '增加用户组', 'usergroup_add', null, null, 'add', 'add', '1', '0', null, '1', null, '2024-05-16 14:18:38', null);
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('38', '23', '10', null, '编辑用户组', 'usergroup_edit', null, null, 'edit', 'edit', '1', '0', null, '1', null, '2024-05-16 14:19:05', null);
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('39', '23', '10', null, '删除用户组', 'usergroup_del', null, null, 'del', 'del', '4', '0', null, '1', null, '2024-05-16 14:19:40', null);
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('40', '23', '10', null, '关联角色', 'associated_role', null, null, 'associated', 'associated', '1', '0', null, '1', null, '2024-05-16 14:20:21', null);
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('50', '25', '1', null, '菜单管理', 'menu_mgr', '/v1/api/menu/**', null, 'ALL', null, '1', '0', null, '1', '1', '2024-04-28 17:11:49', '2024-05-21 11:04:58');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('51', '25', '1', null, '用户组', 'user_group', '/v1/api/userGroup/**', null, 'ALL', null, '2', '0', null, '1', '1', '2024-04-28 17:18:31', '2024-05-21 11:06:08');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('52', '25', '1', null, 'APP配置', 'app_config', '/v1/api/app/config', null, 'ALL', null, '3', '0', null, '1', '1', '2024-04-28 17:19:26', '2024-04-28 17:19:49');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('53', '25', '1', null, '用户管理', 'user_api', '/v1/api/users/**', null, 'ALL', null, '4', '0', null, '1', '1', '2024-04-28 17:20:37', '2024-05-21 11:05:34');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('54', '25', '1', null, '审核记录', 'process_record_api', '/v1/api/biz/process/**', null, 'ALL', null, '4', '0', null, '1', '1', '2024-04-28 17:21:56', '2024-06-14 10:18:04');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('55', '25', '1', null, '任务', 'tasks_api', '/v1/api/biz/tasks/**', null, 'GET,POST,PUT,PATCH,DELETE', null, '6', '0', null, '1', '1', '2024-04-28 17:22:42', '2024-05-21 10:23:28');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('56', '25', '1', null, '船舶分组', 'boat_group', '/v1/api/biz/group/**', null, 'ALL', null, '10', '0', null, '1', '1', '2024-04-28 17:23:37', '2024-04-28 17:23:46');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('57', '25', '1', null, '记录标签', 'marklabel', '/v1/api/biz/marklabel', null, 'ALL', null, '10', '0', null, '1', '1', '2024-04-28 17:24:30', '2024-04-28 17:24:41');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('58', '25', '1', null, '用户JOBS', 'user_jobjs', '/v1/api/biz/job/*', null, 'ALL', null, '9', '0', '', '1', '1', '2024-04-28 17:25:40', '2024-05-21 14:49:39');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('59', '25', '1', null, '船管理', 'boat', '/v1/api/biz/boat', null, 'ALL', null, '10', '0', null, '1', '1', '2024-04-29 09:45:08', '2024-04-29 09:46:22');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('60', '25', '1', null, '规则管理', 'biz_rule', '/v1/api/biz/rule', null, 'ALL', null, '11', '0', null, '1', '1', '2024-04-29 09:47:43', '2024-04-29 09:47:52');
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('61', '25', '1', null, '资源', 'sys_resource', '/v1/api/res/**', null, 'ALL', null, '1', '0', null, '1', null, '2024-04-29 09:48:57', null);
INSERT INTO `sys_menu` (`id`,`parent_id`,`category`,`icon`,`name`,`code`,`url`,`component`,`methods`,`permission_key`,`order`,`status`,`remark`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('62', '25', '1', null, '角色', 'sys_role', '/v1/api/role/**', null, 'ALL', null, '4', '0', null, '1', '1', '2024-04-29 16:36:33', '2024-05-21 11:19:32');
