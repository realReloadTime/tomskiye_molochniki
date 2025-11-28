
namespace BackendSite.Models
{
    public class Statistic
    {
        public int Id { get; set; }
        public string Name { get; set; } = string.Empty;
        public DateTime AnalysisDate { get; set; }
        public int Positive {  get; set; }
        public int Negative { get; set; }
        public int Neutral { get; set; }

        public int UserId { get; set; }
        public User User { get; set; } = null!;
    }
}