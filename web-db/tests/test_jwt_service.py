"""Tests for jwt_service module"""
import pytest
import datetime
from unittest.mock import patch
from co6co_web_db.services.jwt_service import (
    JWT_service,
    createToken,
    decodeToken,
    validToken,
    setCurrentUser,
)


class TestJWT_service:
    """Test cases for JWT_service class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.secret = "test_secret_key"
        self.issuer = "test_issuer"
        self.jwt_svc = JWT_service(self.secret, self.issuer)

    def test_init_with_default_issuer(self):
        """Test JWT_service initialization with default issuer"""
        svc = JWT_service("my_secret")
        assert svc._secret == "my_secret"
        assert svc._issuer == "JWT+SERVICE"

    def test_init_with_custom_issuer(self):
        """Test JWT_service initialization with custom issuer"""
        svc = JWT_service("my_secret", "custom_issuer")
        assert svc._secret == "my_secret"
        assert svc._issuer == "custom_issuer"

    def test_encode_returns_string(self):
        """Test that encode returns a string token"""
        data = {"user_id": 123}
        token = self.jwt_svc.encode(data)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_encode_with_custom_expire_seconds(self):
        """Test encode with custom expiration time"""
        data = {"user_id": 456}
        token = self.jwt_svc.encode(data, expire_seconds=3600)
        assert isinstance(token, str)

    def test_decode_valid_token(self):
        """Test decoding a valid token"""
        data = {"user_id": 789, "name": "test"}
        token = self.jwt_svc.encode(data)
        decoded = self.jwt_svc.decode(token)
        assert decoded is not None
        assert decoded["data"]["user_id"] == 789
        assert decoded["data"]["name"] == "test"

    def test_decode_invalid_token(self):
        """Test decoding an invalid token returns None"""
        invalid_token = "invalid.token.here"
        result = self.jwt_svc.decode(invalid_token)
        assert result is None

    def test_decode_none_token(self):
        """Test decoding None token returns None"""
        result = self.jwt_svc.decode(None)
        assert result is None

    def test_decode_empty_token(self):
        """Test decoding empty token returns None"""
        result = self.jwt_svc.decode("")
        assert result is None


class TestModuleFunctions:
    """Test module-level functions"""

    @pytest.mark.asyncio
    async def test_createToken_returns_string(self):
        """Test createToken returns a string token"""
        data = {"user_id": 100}
        token = await createToken("secret123", data)
        assert isinstance(token, str)
        assert len(token) > 0

    @pytest.mark.asyncio
    async def test_createToken_with_custom_expiration(self):
        """Test createToken with custom expiration"""
        data = {"user_id": 200}
        token = await createToken("secret123", data, expire_seconds=7200)
        assert isinstance(token, str)

    @pytest.mark.asyncio
    async def test_decodeToken_valid_token(self):
        """Test decodeToken with valid token"""
        data = {"user_id": 300}
        token = await createToken("secret456", data)
        result = decodeToken(token, "secret456")
        assert result is not None
        assert result["user_id"] == 300

    @pytest.mark.asyncio
    async def test_decodeToken_invalid_secret(self):
        """Test decodeToken with wrong secret returns None"""
        data = {"user_id": 400}
        token = await createToken("secret789", data)
        result = decodeToken(token, "wrong_secret")
        assert result is None

    def test_decodeToken_none_token(self):
        """Test decodeToken with None token returns None"""
        result = decodeToken(None, "any_secret")
        assert result is None

    def test_decodeToken_empty_token(self):
        """Test decodeToken with empty token returns None"""
        result = decodeToken("", "any_secret")
        assert result is None

    def test_decodeToken_missing_data_field(self):
        """Test decodeToken when token has no data field"""
        import datetime
        import jwt

        dic = {
            'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=86400),
            'iat': datetime.datetime.now(datetime.timezone.utc),
            'iss': "test",
        }
        token = jwt.encode(dic, "secret", algorithm="HS256")
        result = decodeToken(token, "secret")
        assert result is None


class TestSetCurrentUser:
    """Test setCurrentUser function"""

    @pytest.mark.asyncio
    async def test_setCurrentUser_sets_ctx_current_user(self):
        """Test setCurrentUser sets current_user in request context"""
        from unittest.mock import MagicMock

        mock_request = MagicMock(spec=['ctx'])
        mock_request.ctx = MagicMock()
        data = {"user_id": 500, "name": "testuser"}

        result = await setCurrentUser(mock_request, data)

        assert result is True
        assert mock_request.ctx.current_user == data


class TestValidToken:
    """Test validToken function"""

    @pytest.mark.asyncio
    async def test_validToken_with_valid_token(self):
        """Test validToken returns True for valid token"""
        from unittest.mock import MagicMock, AsyncMock, patch

        mock_request = MagicMock()
        mock_request.token = await createToken("secret_key", {"user_id": 600})

        with patch("co6co_web_db.services.jwt_service.setCurrentUser", new_callable=AsyncMock) as mock_set_user:
            mock_set_user.return_value = True
            result = await validToken(mock_request, "secret_key")
            assert result is True

    @pytest.mark.asyncio
    async def test_validToken_with_invalid_token(self):
        """Test validToken returns False for invalid token"""
        from unittest.mock import MagicMock

        mock_request = MagicMock()
        mock_request.token = "invalid_token"

        result = await validToken(mock_request, "secret_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_validToken_with_none_token(self):
        """Test validToken returns False for None token"""
        from unittest.mock import MagicMock

        mock_request = MagicMock()
        mock_request.token = None

        result = await validToken(mock_request, "secret_key")
        assert result is False