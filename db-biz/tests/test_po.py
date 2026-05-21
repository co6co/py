from model.pos.right import UserPO


def test_user_rules():
    user = UserPO()
    us = []
    print("普通list", us is None, len(us), type(us))
    assert len(user.rolePOs) == 0
    assert user.rolePOs is not None
    print("角色list", user.rolePOs is None, len(user.rolePOs), type(user.rolePOs))

    assert user.userGroupPO is None
    print("用户组", user.userGroupPO, type(user.userGroupPO))
