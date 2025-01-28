from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, random_split
import numpy as np

class Neo4jQueryDataset(Dataset):
    def __init__(self, input_texts, target_texts, tokenizer):
        self.encodings = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
        self.labels = tokenizer(target_texts, padding=True, truncation=True, return_tensors="pt")["input_ids"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }

# Prepare training data with common Neo4j query patterns
input_texts = [
    "Find all users who have made purchases",
    "Get all movies released in 2020",
    "Find users who liked action movies",
    "Show me all products with price greater than 100",
    "Find connections between users who are friends",
    "Get users who purchased products in the last 30 days",
    "Find movies directed by Christopher Nolan",
    "Show all users who reviewed restaurants",
    "Find products in the electronics category",
    "Get users who live in New York",
    "Find mutual friends between two users",
    "Show orders with total amount greater than 1000",
    "Find users who both liked the same movie",
    "Get all comments on posts from last week",
    "Find restaurants with rating above 4.5",
    
    # New organizational queries
    "Find all employees in the Engineering department",
    "Show managers who have more than 5 direct reports",
    "List employees with salary greater than 100000",
    "Get departments with total salary budget exceeding 1000000",
    "Find employees who joined in 2023",
    "Show all projects assigned to the Marketing team",
    "List employees who report to Sarah Johnson",
    "Find departments with no manager assigned",
    "Get employees who have completed compliance training",
    "Show all leave requests pending approval",
    "Find employees with performance rating above 4",
    "List departments sorted by average employee salary",
    "Show employees who have worked on both Project X and Project Y",
    "Get departments with more than 20 employees",
    "Find employees who haven't taken vacation in 6 months",
    "Show managers and their department's total headcount",
    "List employees with certifications expiring this month",
    "Find projects with budget overrun",
    "Get employees who have received bonuses in 2023",
    "Show departments with highest turnover rate",
    "Find employees eligible for promotion",
    "List all skills possessed by Engineering team members",
    "Show projects due in the next 30 days",
    "Get employees who worked overtime last month",
    "Find departments under budget this quarter",
    "Show employees with specific certification",
    "List all remote workers in the company",
    "Find managers who also work on projects",
    "Get employees involved in cross-department projects",
    "Show departments with open positions",
    "Find employees who can speak multiple languages",
    "List projects with highest priority",
    "Show employees who have completed leadership training",
    "Get departments with international teams",
    "Find employees who have changed roles internally",
    "Show projects requiring security clearance",
    "List employees with perfect attendance",
    "Find departments with highest training completion rate",
    "Get employees who mentor junior staff",
    "Show teams with all members certified",
    "Find employees with specific technical skills",
    "List departments by gender diversity ratio",
    "Show projects with external contractors",
    "Get employees who have filed HR complaints",
    "Find departments exceeding quality metrics",
    "Show employees with dual department roles",
    "List projects approaching deadline",
    "Find employees eligible for retirement",
    "Get departments with highest customer satisfaction",
    "Show managers with highest team performance",
    "List employees on parental leave",
    "Find projects needing additional resources",
    "Get employees with most training hours",
    "Show departments with best safety records",
    "Find employees who are project leads",
    "List departments by equipment budget",
    "Show employees with specialized certifications",
    "Get projects in testing phase",
    "Find departments with most innovation patents",
    "Show employees who travel frequently",
    "List projects by client satisfaction rating",
    "Find employees with security clearance",
    "Get departments with flexible work hours",
    "Show managers with highest retention rate",
    "List employees by years of service",
    "Find projects using specific technology",
    "Get employees with signing bonuses",
    "Show departments by office location",
    "Find employees with specific degrees",
    "List projects by revenue generation",
    "Show teams with best collaboration scores",
    "Get employees who are subject matter experts",
    "Find departments with most training budget",
    "Show projects with regulatory requirements",
    "List employees by performance improvement",
    "Find managers who were promoted internally",
    "Get departments with sustainability initiatives",
    "Show employees with international experience",
    "List projects by risk assessment level",
    "Find departments with most overtime hours",
    "Get employees with leadership potential",
    "Show teams by productivity metrics",
    "Find projects requiring compliance review",
    "List employees by certification level",
    "Show departments with highest growth rate",
    "Get managers with cross-functional teams",
    "Find employees on performance improvement plans",
    "List projects by resource utilization",
    "Show departments by employee satisfaction",
    "Get employees with specific achievements",
    "Find teams exceeding sales targets",
    "List projects by completion rate",
    "Show employees with perfect safety record",
    "Get departments by training effectiveness",
    "Find managers with most diverse teams",
    "List employees by skill gap analysis",
    "Show projects with highest ROI",
    "Get departments by workplace incidents",
    "Find employees eligible for sabbatical",
    "List teams by innovation metrics",
    "Show projects requiring security audit",
    "Get employees with most client feedback",
    "Find departments by budget efficiency",
    "Show managers with best work-life balance scores",
    "List employees by professional development",
    "Find projects in maintenance phase",
    "Get departments by employee retention",
    "Show teams with highest quality metrics"
]

target_texts = [
    "MATCH (u:User)-[:MADE]->(p:Purchase) RETURN u",
    "MATCH (m:Movie) WHERE m.release_year = 2020 RETURN m",
    "MATCH (u:User)-[:LIKED]->(m:Movie) WHERE m.genre = 'Action' RETURN u",
    "MATCH (p:Product) WHERE p.price > 100 RETURN p",
    "MATCH (u1:User)-[:FRIEND]-(u2:User) RETURN u1, u2",
    "MATCH (u:User)-[:MADE]->(p:Purchase) WHERE p.date >= datetime().minus(duration('P30D')) RETURN u",
    "MATCH (m:Movie)<-[:DIRECTED]-(d:Director {name: 'Christopher Nolan'}) RETURN m",
    "MATCH (u:User)-[:REVIEWED]->(r:Restaurant) RETURN u, r",
    "MATCH (p:Product {category: 'Electronics'}) RETURN p",
    "MATCH (u:User {city: 'New York'}) RETURN u",
    "MATCH (u1:User)-[:FRIEND]-(mutual:User)-[:FRIEND]-(u2:User) WHERE id(u1) < id(u2) RETURN mutual",
    "MATCH (o:Order) WHERE o.total > 1000 RETURN o",
    "MATCH (u1:User)-[:LIKED]->(m:Movie)<-[:LIKED]-(u2:User) WHERE id(u1) < id(u2) RETURN u1, u2, m",
    "MATCH (c:Comment)-[:ON]->(p:Post) WHERE c.date >= datetime().minus(duration('P7D')) RETURN c",
    "MATCH (r:Restaurant) WHERE r.rating > 4.5 RETURN r",
    
    # New organizational Cypher queries
    "MATCH (e:Employee)-[:BELONGS_TO]->(d:Department {name: 'Engineering'}) RETURN e",
    "MATCH (m:Employee)-[:MANAGES]->(e:Employee) WITH m, count(e) as reports WHERE reports > 5 RETURN m",
    "MATCH (e:Employee) WHERE e.salary > 100000 RETURN e",
    "MATCH (d:Department)<-[:BELONGS_TO]-(e:Employee) WITH d, sum(e.salary) as budget WHERE budget > 1000000 RETURN d",
    "MATCH (e:Employee) WHERE e.joinDate >= date('2023-01-01') RETURN e",
    "MATCH (p:Project)-[:ASSIGNED_TO]->(d:Department {name: 'Marketing'}) RETURN p",
    "MATCH (e:Employee)-[:REPORTS_TO]->(m:Employee {name: 'Sarah Johnson'}) RETURN e",
    "MATCH (d:Department) WHERE NOT (d)<-[:MANAGES]-() RETURN d",
    "MATCH (e:Employee)-[:COMPLETED]->(t:Training {type: 'Compliance'}) RETURN e",
    "MATCH (l:LeaveRequest {status: 'Pending'}) RETURN l",
    "MATCH (e:Employee) WHERE e.performanceRating > 4 RETURN e",
    "MATCH (d:Department)<-[:BELONGS_TO]-(e:Employee) WITH d, avg(e.salary) as avgSalary RETURN d ORDER BY avgSalary DESC",
    "MATCH (e:Employee)-[:WORKS_ON]->(p1:Project {name: 'Project X'}), (e)-[:WORKS_ON]->(p2:Project {name: 'Project Y'}) RETURN e",
    "MATCH (d:Department)<-[:BELONGS_TO]-(e:Employee) WITH d, count(e) as empCount WHERE empCount > 20 RETURN d",
    "MATCH (e:Employee)-[:TOOK_VACATION]->(v:Vacation) WHERE v.date >= datetime().minus(duration('P180D')) WITH e WHERE count(v) = 0 RETURN e",
    "MATCH (m:Employee)-[:MANAGES]->(d:Department)<-[:BELONGS_TO]-(e:Employee) WITH m, d, count(e) as headcount RETURN m, d, headcount",
    "MATCH (e:Employee)-[:HAS_CERT]->(c:Certification) WHERE c.expiryDate <= date().plus(duration('P30D')) RETURN e, c",
    "MATCH (p:Project) WHERE p.actualCost > p.budgetedCost RETURN p",
    "MATCH (e:Employee)-[:RECEIVED]->(b:Bonus) WHERE b.year = 2023 RETURN e",
    "MATCH (d:Department)<-[:BELONGS_TO]-(e:Employee) WHERE e.terminationDate IS NOT NULL WITH d, count(e) as turnover RETURN d ORDER BY turnover DESC",
    "MATCH (e:Employee) WHERE e.yearsInRole >= 2 AND e.performanceRating >= 4 RETURN e",
    "MATCH (e:Employee)-[:HAS_SKILL]->(s:Skill)<-[:REQUIRES]-(d:Department {name: 'Engineering'}) RETURN DISTINCT s",
    "MATCH (p:Project) WHERE p.dueDate <= date().plus(duration('P30D')) RETURN p",
    "MATCH (e:Employee)-[:WORKED]->(t:TimeSheet) WHERE t.month = date().month(-1) AND t.overtime > 0 RETURN e",
    "MATCH (d:Department) WHERE d.actualSpend <= d.budgetedSpend RETURN d",
    "MATCH (e:Employee)-[:HAS_CERT]->(c:Certification {name: 'PMP'}) RETURN e",
    "MATCH (e:Employee {isRemote: true}) RETURN e",
    "MATCH (e:Employee)-[:MANAGES]->(d:Department) WHERE EXISTS((e)-[:WORKS_ON]->(:Project)) RETURN e",
    "MATCH (e:Employee)-[:WORKS_ON]->(p:Project) WHERE NOT (e)-[:BELONGS_TO]->(:Department)<-[:ASSIGNED_TO]-(p) RETURN e",
    "MATCH (d:Department)-[:HAS_POSITION]->(p:Position {status: 'Open'}) RETURN d, p",
    "MATCH (e:Employee)-[:SPEAKS]->(l:Language) WITH e, count(l) as langs WHERE langs > 1 RETURN e",
    "MATCH (p:Project) WHERE p.priority = 'High' RETURN p",
    "MATCH (e:Employee)-[:COMPLETED]->(t:Training {type: 'Leadership'}) RETURN e",
    "MATCH (d:Department)-[:HAS_OFFICE]->(o:Office) WHERE o.country <> 'USA' RETURN d",
    "MATCH (e:Employee)-[:HAD_ROLE]->(r:Role) WITH e, count(r) as roles WHERE roles > 1 RETURN e",
    "MATCH (p:Project) WHERE p.securityClearance = true RETURN p",
    "MATCH (e:Employee) WHERE NOT EXISTS((e)-[:ABSENT]->()) RETURN e",
    "MATCH (d:Department)<-[:BELONGS_TO]-(e:Employee)-[:COMPLETED]->(t:Training) WITH d, count(t) as completed, count(e) as total WHERE (1.0 * completed / total) > 0.9 RETURN d",
    "MATCH (e:Employee)-[:MENTORS]->() RETURN e",
    "MATCH (t:Team)<-[:MEMBER_OF]-(e:Employee)-[:HAS_CERT]->() WITH t, count(DISTINCT e) as certified, count(e) as total WHERE certified = total RETURN t",
    "MATCH (e:Employee)-[:HAS_SKILL]->(s:Skill {category: 'Technical'}) RETURN e",
    "MATCH (d:Department)<-[:BELONGS_TO]-(e:Employee) WITH d, count(CASE WHEN e.gender = 'F' THEN 1 END) * 1.0 / count(e) as ratio RETURN d ORDER BY ratio DESC",
    "MATCH (p:Project)-[:USES_CONTRACTOR]->() RETURN p",
    "MATCH (e:Employee)-[:FILED]->(c:Complaint {department: 'HR'}) RETURN e",
    "MATCH (d:Department) WHERE d.qualityScore > d.qualityTarget RETURN d",
    "MATCH (e:Employee)-[:BELONGS_TO]->(d1:Department), (e)-[:BELONGS_TO]->(d2:Department) WHERE id(d1) < id(d2) RETURN e",
    "MATCH (p:Project) WHERE p.deadline <= date().plus(duration('P14D')) RETURN p",
    "MATCH (e:Employee) WHERE e.age >= 60 OR e.yearsOfService >= 30 RETURN e",
    "MATCH (d:Department) WHERE d.customerSatisfaction >= 4.5 RETURN d",
    "MATCH (m:Employee)-[:MANAGES]->(t:Team) WHERE t.performanceScore >= 4.5 RETURN m",
    "MATCH (e:Employee)-[:ON_LEAVE]->(l:Leave {type: 'Parental'}) WHERE l.endDate > date() RETURN e",
    "MATCH (p:Project) WHERE p.resourceUtilization > 90 RETURN p",
    "MATCH (e:Employee)-[:COMPLETED]->(t:Training) WITH e, count(t) as hours ORDER BY hours DESC LIMIT 10 RETURN e",
    "MATCH (d:Department) WHERE d.safetyIncidents = 0 RETURN d",
    "MATCH (e:Employee)-[:LEADS]->(:Project) RETURN e",
    "MATCH (d:Department) WITH d, sum(d.equipmentBudget) as budget ORDER BY budget DESC RETURN d",
    "MATCH (e:Employee)-[:HAS_CERT]->(c:Certification {level: 'Advanced'}) RETURN e",
    "MATCH (p:Project) WHERE p.phase = 'Testing' RETURN p",
    "MATCH (d:Department)-[:FILED]->(p:Patent) WITH d, count(p) as patents ORDER BY patents DESC RETURN d",
    "MATCH (e:Employee)-[:TRAVELS]->() WITH e, count(*) as trips WHERE trips > 10 RETURN e",
    "MATCH (p:Project) WHERE p.clientSatisfaction >= 4 RETURN p",
    "MATCH (e:Employee) WHERE e.securityClearance >= 'Secret' RETURN e",
    "MATCH (d:Department) WHERE d.flexibleHours = true RETURN d",
    "MATCH (m:Employee)-[:MANAGES]->(t:Team) WITH m, count(*) as retained, count(CASE WHEN t.terminated = false THEN 1 END) as current WHERE current/retained > 0.9 RETURN m",
    "MATCH (e:Employee) WITH e ORDER BY e.startDate RETURN e",
    "MATCH (p:Project)-[:USES_TECH]->(t:Technology {name: 'Python'}) RETURN p",
    "MATCH (e:Employee) WHERE e.signingBonus > 0 RETURN e",
    "MATCH (d:Department)-[:LOCATED_IN]->(o:Office) RETURN d, o",
    "MATCH (e:Employee)-[:HAS_DEGREE]->(d:Degree {type: 'PhD'}) RETURN e",
    "MATCH (p:Project) WITH p ORDER BY p.revenue DESC RETURN p",
    "MATCH (t:Team) WHERE t.collaborationScore >= 4.5 RETURN t",
    "MATCH (e:Employee)-[:IS_SME]->(:Domain) RETURN e",
    "MATCH (d:Department) WITH d ORDER BY d.trainingBudget DESC LIMIT 5 RETURN d",
    "MATCH (p:Project) WHERE p.requiresCompliance = true RETURN p",
    "MATCH (e:Employee) WHERE e.currentPerformance > e.lastPerformance RETURN e",
    "MATCH (m:Employee)-[:MANAGES]->() WHERE EXISTS((m)<-[:PROMOTED]-()) RETURN m",
    "MATCH (d:Department)-[:HAS_INITIATIVE]->(:Sustainability) RETURN d",
    "MATCH (e:Employee)-[:WORKED_IN]->(:Country) WITH e, count(*) as countries WHERE countries > 1 RETURN e",
    "MATCH (p:Project) WHERE p.riskLevel >= 'High' RETURN p",
    "MATCH (d:Department)-[:LOGGED]->(h:Hours {type: 'Overtime'}) WITH d, sum(h.amount) as overtime ORDER BY overtime DESC RETURN d",
    "MATCH (e:Employee) WHERE e.leadershipScore >= 4 RETURN e",
    "MATCH (t:Team) WHERE t.productivityScore > t.targetScore RETURN t",
    "MATCH (p:Project)-[:REQUIRES]->(c:ComplianceReview) WHERE c.status = 'Pending' RETURN p",
    "MATCH (e:Employee)-[:HAS_CERT]->(c:Certification) WITH e, count(c) as certs ORDER BY certs DESC RETURN e",
    "MATCH (d:Department) WITH d, d.currentSize / d.lastYearSize as growth WHERE growth > 1.1 RETURN d",
    "MATCH (m:Employee)-[:MANAGES]->(t:Team) WHERE size((t)-[:SPANS]->(:Department)) > 1 RETURN m",
    "MATCH (e:Employee)-[:ON_PIP]->() RETURN e",
    "MATCH (p:Project) WHERE p.resourceUtilization > p.targetUtilization RETURN p",
    "MATCH (d:Department)<-[:BELONGS_TO]-(e:Employee) WITH d, avg(e.satisfaction) as satisfaction ORDER BY satisfaction DESC RETURN d",
    "MATCH (e:Employee)-[:ACHIEVED]->(a:Achievement) WHERE a.year = date().year RETURN e",
    "MATCH (t:Team)-[:HAS_TARGET]->(s:SalesTarget) WHERE t.sales > s.amount RETURN t",
    "MATCH (p:Project) WITH p, p.completedTasks * 1.0 / p.totalTasks as completion WHERE completion > 0.8 RETURN p",
    "MATCH (e:Employee) WHERE NOT EXISTS((e)-[:HAD_ACCIDENT]->()) RETURN e",
    "MATCH (d:Department)-[:CONDUCTED]->(t:Training) WITH d, avg(t.effectiveness) as effectiveness ORDER BY effectiveness DESC RETURN d",
    "MATCH (m:Employee)-[:MANAGES]->(t:Team) WHERE size([(t)-[:HAS_MEMBER]->(e) WHERE e.gender <> t.members[0].gender | e]) > size(t.members) * 0.4 RETURN m",
    "MATCH (e:Employee)-[:NEEDS_SKILL]->(s:Skill) RETURN e, collect(s.name)",
    "MATCH (p:Project) WITH p ORDER BY p.roi DESC LIMIT 10 RETURN p",
    "MATCH (d:Department)-[:HAD_INCIDENT]->(i:Incident) WITH d, count(i) as incidents ORDER BY incidents RETURN d",
    "MATCH (e:Employee) WHERE e.yearsOfService >= 5 AND NOT EXISTS((e)-[:TOOK_SABBATICAL]->()) RETURN e",
    "MATCH (t:Team)-[:SUBMITTED]->(i:Innovation) WITH t, count(i) as innovations ORDER BY innovations DESC RETURN t",
    "MATCH (p:Project) WHERE EXISTS((p)-[:REQUIRES]->(:SecurityAudit)) RETURN p",
    "MATCH (e:Employee)<-[:GIVEN_BY]-(:Feedback) WITH e, count(*) as feedback ORDER BY feedback DESC LIMIT 10 RETURN e",
    "MATCH (d:Department) WHERE d.actualSpend <= d.budgetedSpend * 0.9 RETURN d",
    "MATCH (m:Employee)-[:MANAGES]->(t:Team) WHERE t.workLifeBalance >= 4 RETURN m",
    "MATCH (e:Employee)-[:ATTENDED]->(t:Training) WITH e, count(t) as development ORDER BY development DESC RETURN e",
    "MATCH (p:Project) WHERE p.phase = 'Maintenance' RETURN p",
    "MATCH (d:Department) WITH d, d.retainedEmployees * 1.0 / d.totalEmployees as retention WHERE retention > 0.9 RETURN d",
    "MATCH (t:Team) WHERE t.qualityMetrics >= t.qualityTarget RETURN t"
]

# Duplicate the dataset a few times to have more training examples
input_texts = input_texts * 3
target_texts = target_texts * 3

# Initialize model and tokenizer
model_name = "facebook/bart-base"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Create full dataset
full_dataset = Neo4jQueryDataset(input_texts, target_texts, tokenizer)

# Split into train and validation
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        print(f"Current loss: {loss.item()}")  # Print loss for monitoring
        return (loss, outputs) if return_outputs else loss

# Define training arguments with improved parameters
training_args = TrainingArguments(
    output_dir='./neo4j_query_model',    
    num_train_epochs=10,                 # Increased epochs
    per_device_train_batch_size=4,       
    per_device_eval_batch_size=4,
    warmup_ratio=0.1,                    # Warmup ratio instead of steps
    weight_decay=0.01,                   
    logging_dir='./logs',                
    logging_steps=1,                     # Log every step
    eval_steps=10,                       # Evaluate every 10 steps
    evaluation_strategy="steps",         # Enable evaluation
    save_strategy="steps",               
    save_steps=10,                       
    save_total_limit=2,                  
    learning_rate=5e-5,                  # Slightly higher learning rate
    load_best_model_at_end=True,         # Load the best model at the end
    metric_for_best_model="loss",
    greater_is_better=False,
)

# Initialize Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("Starting training...")
# Fine-tune the model
trainer.train()

print("Training completed. Saving model...")
# Save the model
model.save_pretrained("./neo4j_query_model/final")
tokenizer.save_pretrained("./neo4j_query_model/final")
print("Model saved successfully!")