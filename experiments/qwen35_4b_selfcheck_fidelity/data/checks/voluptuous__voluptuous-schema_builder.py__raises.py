        @contextmanager
        def raises(
            exc, msg: typing.Optional[str] = None, regex: typing.Optional[re.Pattern] = None
        ) -> Generator[None, None, None]:
        