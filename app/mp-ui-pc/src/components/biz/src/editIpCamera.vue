<template>
	<!-- 弹出框 -->
	<el-dialog
		:title="form.title"
		v-model="form.dialogVisible"
		style="width: 50%; height: 80%">
		<el-form
			label-width="90px"
			ref="dialogForm"
			:rules="rules"
			:model="form.fromData"
			style="max-width: 460px">
			<el-form-item label="名称" prop="name">
				<el-input
					v-model="form.fromData.name"
					placeholder="设备名称"></el-input>
			</el-form-item>
			<el-form-item label="所属站点" prop="siteId"> 
				<el-select 
					style="width: 160px"
					:disabled="!allowModifySite"
					class="mr10"
					clearable
					v-model="form.fromData.siteId"
					placeholder="请选择">
					<el-option
						v-for="item in SiteCategoryRef.List"
						:key="item.id"
						:label="item.name"
						:value="item.id" />
				</el-select>
			</el-form-item>

			<el-form-item label="内网IP" prop="innerIp">
				<el-input
					v-model="form.fromData.innerIp"
					placeholder="内外IP"></el-input>
			</el-form-item>
			<el-form-item label="Sip地址" prop="sip">
				<el-input v-model="form.fromData.sip" placeholder="Sip地址"></el-input>
			</el-form-item>
			<el-form-item label="对讲号" prop="talkbackNo">
				<el-input-number
					v-model="form.fromData.talkbackNo"
					placeholder="对讲号"></el-input-number>
			</el-form-item>
			<el-form-item label="通道1" prop="channel1_sip">
				<el-input
					v-model="form.fromData.channel1_sip"
					placeholder="通道1地址"></el-input>
			</el-form-item>
			<el-form-item label="通道2" prop="channel2_sip">
				<el-input
					v-model="form.fromData.channel2_sip"
					placeholder="通道2地址"></el-input>
			</el-form-item>

			<el-form-item label="通道3" prop="channel3_sip">
				<el-input
					v-model="form.fromData.channel3_sip"
					placeholder="通道3地址"></el-input>
			</el-form-item>

			<el-form-item label="通道4" prop="channel4_sip">
				<el-input
					v-model="form.fromData.channel4_sip"
					placeholder="通道4地址"></el-input>
			</el-form-item>

			<el-form-item label="通道5" prop="channel5_sip">
				<el-input
					v-model="form.fromData.channel5_sip"
					placeholder="通道5地址"></el-input>
			</el-form-item>

			<el-form-item label="流信息" class="streamInfo">
				<el-button @click="onGenerateStreamAddress">生成流地址</el-button>
				<el-card
					style="height: 136px"
					v-for="(steam, index) in form.fromData.streamUrls"
					:key="index">
					<el-form-item
						label="流名称"
						:prop="'streamUrls.' + index + '.name'"
						:rules="{
							required: true,
							message: '请输入流名称',
							trigger: 'blur',
						}">
						<el-input
							v-model="steam.name"
							placeholder="视频流名称"
							required></el-input>
					</el-form-item>

					<el-form-item
						label="流地址"
						:prop="'streamUrls.' + index + '.url'"
						:rules="{
							required: true,
							message: '请输入流地址',
							trigger: 'blur',
						}">
						<el-input
							v-model="steam.url"
							placeholder="视频流地址"
							required></el-input>
					</el-form-item>
					<el-button @click="removeStream(index)" :icon="Minus"> </el-button>
				</el-card>
				<el-button @click="addStream" :icon="Plus"></el-button>
			</el-form-item>
		</el-form>
		<template #footer>
			<span class="dialog-footer">
				<el-button @click="form.dialogVisible = false">关闭</el-button>
				<el-button :loading="form.loading" :disabled="form.loading" @click="onDialogSave(dialogForm)">保存</el-button>
			</span>
		</template>
	</el-dialog>
</template>

<script setup lang="ts">
	import { ref, reactive, PropType } from 'vue';
	import {
		ElMessage,
		ElMessageBox,
		FormRules,
		FormInstance,
	} from 'element-plus';
	import { Plus, Minus } from '@element-plus/icons-vue';
	import * as api from '../../../api/device';
	import * as site_api from '../../../api/site';

	const props = defineProps({
		allowModifySite: {
			type: Boolean, 
			default: true,
		},
	});
	interface CameraItem {
		id?: number;
		innerIp: string;
		name: string;
		sip: string;
		channel1_sip: string;
		channel2_sip: string;
		channel3_sip: string;
		channel4_sip: string;
		channel5_sip: string;
		siteId?: number;
		talkbackNo?: number;
		streams: String;
	}
	interface FromData {
		innerIp: string;
		name: string;
		sip: string;
		channel1_sip: string;
		channel2_sip: string;
		channel3_sip: string;
		channel4_sip: string;
		channel5_sip: string;
		siteId?: number;
		talkbackNo?: number;
		streamUrls: Array<{ name: String; url: String }>;
	}

	const emits = defineEmits(['saved']);
	interface dialogDataType {
		dialogVisible: boolean;
		operation: 0 | 1 | number;
		title?: string;
		id: number;
		loading:boolean;
		fromData: FromData;
	}
	const dialogForm = ref<FormInstance>();
	const rules: FormRules = {
		siteId: [{ required: true, message: '请选择所属站点', trigger: ['blur'] }],
		name: [{ required: true, message: '请输入设备名称', trigger: 'blur' }],
		sip: [{ required: true, message: '请输入sip地址', trigger: 'blur' }],
		channel1_sip: [
			{ required: true, len: 20, message: '请输入通道2地址', trigger: 'blur' },
		],
		channel2_sip: [
			{ required: true, len: 20, message: '请输入通道2地址', trigger: 'blur' },
		],
		channel3_sip: [{ len: 20, message: '请输入通道3地址', trigger: 'blur' }],
		channel4_sip: [{ len: 20, message: '请输入通道4地址', trigger: 'blur' }],
		channel5_sip: [{ len: 20, message: '请输入通道5地址', trigger: 'blur' }],
		innerIp: [{ required: true, message: '请输入设备IP', trigger: ['blur'] }],
		talkbackNo: [
			{
				type: 'number',
				required: true,
				message: '对讲号不正确',
				trigger: ['blur'],
			},
		],
		streamName: [
			{ required: true, message: '请视频地址名称', trigger: ['blur'] },
		],
		streamUrl: [{ required: true, message: '请视频地址', trigger: ['blur'] }],
	};
	let form = reactive<dialogDataType>({
		dialogVisible: false,
		operation: 0,
		id: 0,
		loading:false,
		fromData: {
			innerIp: '',
			name: '',
			sip: '',
			channel1_sip: '',
			channel2_sip: '',
			channel3_sip: '',
			channel4_sip: '',
			channel5_sip: '',
			streamUrls: [],
		},
	});
	interface SiteCategory {
		List: Array<{ id: number; name: string }>;
	}
	const SiteCategoryRef = ref<SiteCategory>({ List: [] });
	const getSiteType = async () => {
		const res = await site_api.select_svc();
		if (res.code == 0) {
			SiteCategoryRef.value.List = res.data;
		}
	};
	getSiteType();
	const onOpenDialog = (operation: 0 | 1, item: CameraItem) => {
		form.dialogVisible = true;

		form.dialogVisible = true;
		form.operation = operation;
		form.id = -1;
		switch (operation) {
			case 0:
				form.title = '增加';
				form.fromData.innerIp = '';
				form.fromData.name = '';
				form.fromData.sip = '';
				form.fromData.channel1_sip = '';
				form.fromData.channel2_sip = '';
				form.fromData.channel3_sip = '';
				form.fromData.channel4_sip = '';
				form.fromData.channel5_sip = '';
				form.fromData.siteId = item.siteId;
				form.fromData.talkbackNo = undefined;
				form.fromData.streamUrls = [];
				break;
			case 1:
				if (item && item.id) {
					const row = item;
					form.id = item.id;
					form.title = '编辑';
					form.fromData.innerIp = row.innerIp;
					form.fromData.name = row.name;
					form.fromData.sip = row.sip;
					form.fromData.channel1_sip = row.channel1_sip;
					form.fromData.channel2_sip = row.channel2_sip;
					form.fromData.channel3_sip = row.channel3_sip;
					form.fromData.channel4_sip = row.channel4_sip;
					form.fromData.channel5_sip = row.channel5_sip;
					form.fromData.siteId = row.siteId;
					form.fromData.talkbackNo = row.talkbackNo;
					if (row.streams && typeof row.streams == 'string')
						form.fromData.streamUrls = JSON.parse(row.streams);
					else form.fromData.streamUrls = [];
				}

				break;
		}
	};
	const removeStream = (index: number) => {
		form.fromData.streamUrls.splice(index, 1);
	};
	const addStream = () => {
		form.fromData.streamUrls.push({ name: '', url: '' });
	};
	const onDialogSave = (formEl: FormInstance | undefined) => {
		if (!formEl) return;
		formEl.validate((value) => {
			if (value) {
				form.loading=true
				if (form.operation == 0) {
					api.add_camera_svc(form.fromData).then((res) => {
						if (res.code == 0) {
							form.dialogVisible = false;
							ElMessage.success(`增加成功`);
							emits('saved');
						} else {
							ElMessage.error(`增加失败:${res.message}`);
						}
					}).finally(()=>{form.loading=false});
				} else {
					api.edit_camera_svc(form.id, form.fromData).then((res) => {
						if (res.code == 0) {
							form.dialogVisible = false;
							ElMessage.success(`编辑成功`);
							emits('saved');
						} else {
							ElMessage.error(`编辑失败:${res.message}`);
						}
					}).finally(()=>{form.loading=false});
				}
			} else {
				ElMessage.error('请检查输入的数据！');
				return false;
			}
		});
	};
	//通过通道号生成流地址
	const onGenerateStreamAddress = () => {
		ElMessageBox.confirm(`将删除以前的流地址信息，确定要生成？`, '提示', {
			type: 'warning',
		})
			.then(() => {
				let streamUrl = [];
				let channelArr = [];
				channelArr.push(form.fromData.channel1_sip);
				channelArr.push(form.fromData.channel2_sip);
				channelArr.push(form.fromData.channel3_sip);
				channelArr.push(form.fromData.channel4_sip);
				channelArr.push(form.fromData.channel5_sip);
				for (let i = 0; i < channelArr.length; i++) {
					let value = channelArr[i];
					if (value && value.length == 20) {
						let t = {
							name: '通道' + (i + 1),
							url: `wss://stream.jshwx.com.cn:8441/flv_ws?device=gb${value}&type=rt.flv`,
						};
						streamUrl.push(t);
					}
				}
				if (streamUrl.length == 0) {
					ElMessage.warning('请先输入对应的通道号！');
				}
				form.fromData.streamUrls = streamUrl;
			})
			.catch(() => {});
	};
	defineExpose({
		onOpenDialog,
	});
</script>
