from typing import Generic, TypeVar, Any

_S = TypeVar("_S", bound="Session")


class Session:
    sessionName = None

    def __init__(self, name) -> None:
        print("Session init")
        self.sessionName = name
        pass

    def print(self):
        print("hellow:", self.sessionName)


class sessionmaker(Generic[_S]):
    def __init__(self) -> None:
        print("工厂API")
        super().__init__()

    def __call__(self, **local_kw: Any) -> _S:
        return Session(**local_kw)


S = sessionmaker()
s = S(name="你好")
s.print()
