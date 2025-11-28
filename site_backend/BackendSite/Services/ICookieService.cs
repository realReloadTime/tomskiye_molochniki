namespace BackendSite.Services
{
    public interface ICookieService
    {
        void SetSessionCookie(string sessionToken);
        string GetSessionToken();
        void DeleteSessionCookie();
    }
}