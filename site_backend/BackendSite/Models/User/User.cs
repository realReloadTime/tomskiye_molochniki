
namespace BackendSite.Models
{
    public class User
    {
        public int Id { get; set; } 
        public string Login {  get; set; } = string.Empty;
        public string PasswordHash { get; set; } = string.Empty;
        public string? SessionToken { get; set; } 
        public DateTime? SessionExpires { get; set; } 

        public ICollection<Statistic> HasStatistic { get; set; } = new List<Statistic>();
    }
}