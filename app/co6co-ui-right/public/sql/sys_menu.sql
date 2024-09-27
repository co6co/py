/*
Navicat MySQL Data Transfer

Source Server         : localhost
Source Server Version : 50721
Source Host           : localhost:3306
Source Database       : systemdb

Target Server Type    : MYSQL
Target Server Version : 50721
File Encoding         : 65001

Date: 2024-09-27 11:47:52
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
INSERT INTO `sys_menu` VALUES ('0', null, '0', null, '根节点', 'root', null, null, '', null, '0', '0', '12', '1', '1', '2024-04-28 14:44:24', '2024-04-28 14:44:37');
INSERT INTO `sys_menu` VALUES ('3', '0', '2', 'Avatar', '用户管理', 'user', '/usermgr', '/views/permission/UserTableView.vue', null, 'admin', '1', '0', null, '1', '1', '2024-04-28 16:08:40', '2024-07-01 13:45:18');
INSERT INTO `sys_menu` VALUES ('4', '3', '10', null, '用户增加', 'user_add', null, null, 'add', 'add', '1', '0', null, '1', '1', '2024-04-28 17:06:05', '2024-05-16 09:37:02');
INSERT INTO `sys_menu` VALUES ('5', '3', '10', null, '用户编辑', 'user_edit', null, null, 'edit', 'edit', '2', '0', null, '1', '1', '2024-04-28 17:06:41', '2024-05-16 10:22:36');
INSERT INTO `sys_menu` VALUES ('6', '3', '10', null, '用户删除', 'user_del', null, null, 'del', 'del', '3', '0', null, '1', '1', '2024-04-28 17:07:22', '2024-05-16 09:42:43');
INSERT INTO `sys_menu` VALUES ('7', '3', '10', null, '重置密码', 'user_reset_passwd', null, null, 'reset', 'reset_pwd', '4', '0', null, '1', '1', '2024-04-28 17:08:10', '2024-05-16 09:42:51');
INSERT INTO `sys_menu` VALUES ('8', '3', '3', 'Avatar', '修改当前密码', 'current_user_info', '/user', '../views/user.vue', '', 'user', '10', '0', '', '1', null, '2024-05-21 16:42:29', '2024-05-21 16:44:25');
INSERT INTO `sys_menu` VALUES ('9', '3', '10', null, '用户关联角色', 'user_associated', null, null, 'associated', 'associated', '5', '0', null, '1', null, '2024-05-16 13:52:17', null);
INSERT INTO `sys_menu` VALUES ('20', '0', '0', 'Key', '系统维护', 'SYSTEM', null, null, null, 'admin', '8', '0', null, '1', '1', '2024-04-28 17:02:04', '2024-05-17 15:13:49');
INSERT INTO `sys_menu` VALUES ('21', '20', '2', 'Menu', '系统菜单', 'menu', '/menu', '/views/permission/MenuTreeView.vue', null, 'admin', '1', '0', null, '1', '1', '2024-04-28 17:03:10', '2024-07-01 11:59:41');
INSERT INTO `sys_menu` VALUES ('22', '20', '2', 'ColdDrink', '角色管理', 'role', '/role', '/views/permission/RoleView.vue', null, 'admin', '3', '0', null, '1', '1', '2024-04-29 16:34:19', '2024-07-01 12:00:13');
INSERT INTO `sys_menu` VALUES ('23', '20', '2', 'Avatar', '用户组', 'userGroup', '/userGroup', '/views/permission/UserGroupTreeView.vue', null, 'supper', '2', '0', null, '1', '1', '2024-04-28 17:04:19', '2024-07-01 12:00:05');
INSERT INTO `sys_menu` VALUES ('25', '0', '0', null, 'API管理', 'API_MGR', null, null, null, null, '999', '0', null, '1', '1', '2024-04-28 17:09:27', '2024-04-28 17:13:40');
INSERT INTO `sys_menu` VALUES ('30', '21', '10', null, '增加系统菜单', 'menu_add', null, null, 'add', 'add', '1', '0', null, '1', null, '2024-05-16 11:22:37', null);
INSERT INTO `sys_menu` VALUES ('31', '21', '10', null, '编辑系统菜单', 'menu_edit', null, null, 'edit', 'edit', '2', '0', null, '1', null, '2024-05-16 11:23:24', null);
INSERT INTO `sys_menu` VALUES ('32', '21', '10', null, '删除菜单', 'del_menu', null, null, 'del', 'menu_del', '3', '0', null, '1', null, '2024-05-16 11:24:15', null);
INSERT INTO `sys_menu` VALUES ('33', '22', '10', null, '增加角色', 'role_add', null, null, 'add', 'add', '1', '0', null, '1', '1', '2024-05-16 11:25:22', '2024-05-16 11:27:39');
INSERT INTO `sys_menu` VALUES ('34', '22', '10', null, '编辑角色', 'role_edit', null, null, 'edit', 'edit', '2', '0', null, '1', '1', '2024-05-16 11:26:10', '2024-05-16 11:35:15');
INSERT INTO `sys_menu` VALUES ('35', '22', '10', null, '角色关联菜单', 'role_ass', null, null, 'associated', 'associated', '3', '0', null, '1', '1', '2024-05-16 11:27:17', '2024-05-16 11:35:32');
INSERT INTO `sys_menu` VALUES ('36', '22', '10', null, '删除角色', 'role_del', null, null, 'del', 'del', '4', '0', null, '1', null, '2024-05-16 11:33:43', null);
INSERT INTO `sys_menu` VALUES ('37', '23', '10', null, '增加用户组', 'usergroup_add', null, null, 'add', 'add', '1', '0', null, '1', null, '2024-05-16 14:18:38', null);
INSERT INTO `sys_menu` VALUES ('38', '23', '10', null, '编辑用户组', 'usergroup_edit', null, null, 'edit', 'edit', '1', '0', null, '1', null, '2024-05-16 14:19:05', null);
INSERT INTO `sys_menu` VALUES ('39', '23', '10', null, '删除用户组', 'usergroup_del', null, null, 'del', 'del', '4', '0', null, '1', null, '2024-05-16 14:19:40', null);
INSERT INTO `sys_menu` VALUES ('40', '23', '10', null, '关联角色', 'associated_role', null, null, 'associated', 'associated', '1', '0', null, '1', null, '2024-05-16 14:20:21', null);
INSERT INTO `sys_menu` VALUES ('50', '25', '1', null, '菜单API', 'menu_mgr', '/v1/api/menu/**', null, 'ALL', null, '1', '0', null, '1', '1', '2024-04-28 17:11:49', '2024-09-27 11:36:51');
INSERT INTO `sys_menu` VALUES ('51', '25', '1', null, '用户组API', 'user_group', '/v1/api/userGroup/**', null, 'ALL', null, '2', '0', null, '1', '1', '2024-04-28 17:18:31', '2024-09-27 11:37:05');
INSERT INTO `sys_menu` VALUES ('52', '25', '1', null, 'APP配置API', 'app_config', '/v1/api/app/config', null, 'ALL', null, '3', '0', null, '1', '1', '2024-04-28 17:19:26', '2024-09-27 11:37:22');
INSERT INTO `sys_menu` VALUES ('53', '25', '1', null, '用户API', 'user_api', '/v1/api/users/**', null, 'ALL', null, '4', '0', null, '1', '1', '2024-04-28 17:20:37', '2024-09-27 11:37:40');
INSERT INTO `sys_menu` VALUES ('61', '25', '1', null, '资源API', 'sys_resource', '/v1/api/res/**', null, 'ALL', null, '1', '0', null, '1', '1', '2024-04-29 09:48:57', '2024-09-27 11:36:43');
INSERT INTO `sys_menu` VALUES ('62', '25', '1', null, '角色API', 'sys_role', '/v1/api/role/**', null, 'ALL', null, '4', '0', null, '1', '1', '2024-04-29 16:36:33', '2024-09-27 11:37:47');
INSERT INTO `sys_menu` VALUES ('63', '20', '2', 'Setting', '系统配置', 'system_config', '/configs', '/views/permission/ConfigView.vue', null, 'admin', '1', '0', null, '1', null, '2024-07-18 14:09:34', null);
INSERT INTO `sys_menu` VALUES ('64', '63', '10', null, '增加', 'system_config_add', null, null, 'add', 'add', '1', '0', null, '1', null, '2024-07-18 14:10:27', null);
INSERT INTO `sys_menu` VALUES ('65', '63', '10', null, '编辑', 'system_config_edit', null, null, 'edit', 'edit', '2', '0', null, '1', null, '2024-07-18 14:11:12', null);
INSERT INTO `sys_menu` VALUES ('66', '63', '10', null, '删除', 'system_config_del', null, null, 'del', 'del', '3', '0', null, '1', null, '2024-07-18 14:12:05', null);
INSERT INTO `sys_menu` VALUES ('67', '20', '2', 'Coin', '字典管理', 'sys_dict_type', '/sysdictType', '/views/permission/DictTypeView.vue', null, 'admin', '2', '0', null, '1', '1', '2024-07-18 14:16:32', '2024-07-18 15:52:47');
INSERT INTO `sys_menu` VALUES ('68', '67', '3', 'Grid', '字典', 'sys_dict', '/systemdict/:id', '/views/permission/DictView.vue', 'view', 'view', '1', '0', null, '1', '1', '2024-07-18 14:19:36', '2024-07-18 17:33:40');
INSERT INTO `sys_menu` VALUES ('69', '67', '10', null, '增加', 'system_dict_type_add', null, null, 'add', 'add', '2', '0', null, '1', '1', '2024-07-18 14:20:27', '2024-07-19 17:25:51');
INSERT INTO `sys_menu` VALUES ('70', '67', '10', null, '编辑', 'sys_dict_type_edit', null, null, 'edit', 'edit', '3', '0', null, '1', null, '2024-07-18 14:21:01', null);
INSERT INTO `sys_menu` VALUES ('71', '67', '10', null, '删除', 'sys_dict_type_del', null, null, 'del', 'del', '4', '0', null, '1', null, '2024-07-18 14:21:30', null);
INSERT INTO `sys_menu` VALUES ('72', '68', '10', null, '增加', 'sys_dict_add', null, null, 'add', 'add', '1', '0', null, '1', null, '2024-07-18 14:30:30', null);
INSERT INTO `sys_menu` VALUES ('73', '68', '10', null, '编辑', 'sys_dict_edit', null, null, 'edit', 'edit', '2', '0', null, '1', null, '2024-07-18 14:31:07', null);
INSERT INTO `sys_menu` VALUES ('74', '68', '10', null, '删除', 'sys_dict_del', null, null, 'del', 'del', '3', '0', null, '1', '1', '2024-07-18 14:31:31', '2024-07-18 14:32:01');
INSERT INTO `sys_menu` VALUES ('75', '25', '1', null, '配置API', 'sys_config_api', '/v1/api/config/**', null, 'ALL', null, '1', '0', null, '1', '1', '2024-07-18 15:50:13', '2024-09-27 11:36:35');
INSERT INTO `sys_menu` VALUES ('76', '25', '1', null, '字典API', 'sys_dict_api', '/v1/api/dict/**', null, 'ALL', null, '1', '0', null, '1', '1', '2024-07-18 15:50:48', '2024-09-27 11:37:00');
